"""
NeuroFlux Checkpoint Management
==============================
Implements fault-tolerant checkpointing with RAID integration from Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from neuroflux.system.raid import EnhancedRAID6
import threading
from neuroflux.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class CheckpointManager:
    """
    Enhanced checkpoint management with RAID integration and compression
    Implements fault-tolerant checkpointing from Section 2.2
    """
    def __init__(
        self,
        save_dir: str = './checkpoints',
        max_checkpoints: int = 5,
        compression_level: int = 3,
        raid: Optional[EnhancedRAID6] = None
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compression_level = compression_level
        self.checkpoint_list = []
        self.raid = raid
        
        # Add lock for thread safety
        self.raid_lock = threading.Lock()
        self.checkpoint_lock = threading.Lock()
        
        # Initialize error tracking
        self.error_counts = torch.zeros(max_checkpoints)
        self.last_verified_step = 0
        
    def compress_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compress model weights to FP8 where safe"""
        compressed = {}
        for key, tensor in state_dict.items():
            # Keep certain layers in FP16/32 for stability
            if any(s in key for s in ['norm', 'embed', 'head']):
                compressed[key] = tensor
            else:
                compressed[key] = tensor.to(torch.float8_e4m3fn)
        return compressed
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        hypernet: nn.Module,
        raid_memory: Optional[Dict] = None,
        step: int = 0,
        metrics: Optional[Dict] = None,
        verify: bool = True
    ) -> str:
        """Thread-safe checkpoint saving"""
        with self.checkpoint_lock:
            return self._save(
                model,
                optimizer,
                scheduler,
                hypernet,
                raid_memory,
                step,
                metrics,
                verify
            )
    
    def _save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        hypernet: nn.Module,
        raid_memory: Optional[Dict] = None,
        step: int = 0,
        metrics: Optional[Dict] = None,
        verify: bool = True
    ) -> str:
        """Save checkpoint with compression and RAID state"""
        checkpoint_path = self.save_dir / f'checkpoint_{step}.pt'
        
        try:
            # Compress model state
            model_state = self.compress_state_dict(model.state_dict())
            
            checkpoint = {
                'step': step,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler else None,
                'hypernet_state': hypernet.state_dict(),
                'raid_memory': raid_memory,
                'metrics': metrics
            }
            
            # Save with compression
            torch.save(
                checkpoint,
                checkpoint_path,
                _use_new_zipfile_serialization=True,
                compression=self.compression_level
            )
            
            # Verify if requested
            if verify and not self.verify_integrity(checkpoint_path):
                raise RuntimeError("Checkpoint verification failed")
            
            # Manage checkpoint history
            self.checkpoint_list.append(checkpoint_path)
            if len(self.checkpoint_list) > self.max_checkpoints:
                oldest = self.checkpoint_list.pop(0)
                oldest.unlink()
                
            self.last_verified_step = step
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {str(e)}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        hypernet: Optional[nn.Module] = None,
        device: str = 'cuda',
        verify: bool = True
    ) -> Tuple[int, Optional[Dict]]:
        """Load checkpoint with automatic device placement"""
        if verify and not self.verify_integrity(checkpoint_path):
            raise RuntimeError("Checkpoint integrity verification failed")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        try:
            # Load model with FP8 -> FP16/32 conversion
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if scheduler and checkpoint['scheduler_state']:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                
            if hypernet and checkpoint['hypernet_state']:
                hypernet.load_state_dict(checkpoint['hypernet_state'])
                
            # Restore RAID memory if available
            if self.raid and checkpoint.get('raid_memory'):
                self.raid.restore_memory(checkpoint['raid_memory'])
                
            return checkpoint['step'], checkpoint.get('metrics')
            
        except Exception as e:
            logger.error(f"Checkpoint load failed: {str(e)}")
            raise
    
    def find_latest(self) -> Optional[str]:
        """Find most recent checkpoint"""
        checkpoints = sorted(self.save_dir.glob('checkpoint_*.pt'))
        return str(checkpoints[-1]) if checkpoints else None
    
    def verify_integrity(self, checkpoint_path: str) -> bool:
        """Thread-safe checkpoint verification"""
        with self.raid_lock:
            try:
                checkpoint = torch.load(checkpoint_path)
                
                # Verify model state
                if self.raid and not self.raid.verify_data_integrity(
                    checkpoint['model_state']
                ):
                    return False
                    
                # Verify RAID memory if present
                if self.raid and checkpoint.get('raid_memory'):
                    if not self.raid.verify_memory_integrity(
                        checkpoint['raid_memory']
                    ):
                        return False
                        
                return True
            except Exception as e:
                logger.error(f"Checkpoint verification failed: {str(e)}")
                return False
    
    def recover_from_corruption(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        hypernet: nn.Module
    ) -> bool:
        """Attempt to recover from corrupted checkpoint"""
        latest = self.find_latest()
        if not latest:
            return False
            
        try:
            # Attempt RAID recovery first
            if self.raid and self.raid.can_recover():
                logger.info("Attempting RAID recovery...")
                self.raid.rebuild_parity()
                return True
                
            # Fall back to previous checkpoint
            logger.info("Falling back to previous checkpoint...")
            prev_checkpoints = sorted(self.save_dir.glob('checkpoint_*.pt'))[:-1]
            for checkpoint_path in reversed(prev_checkpoints):
                if self.verify_integrity(checkpoint_path):
                    self.load(
                        checkpoint_path,
                        model,
                        optimizer,
                        hypernet=hypernet
                    )
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            return False
    
    def get_adaptive_interval(self) -> int:
        """Get adaptive checkpoint interval based on error rates"""
        if self.raid:
            return self.raid.get_adaptive_checkpoint_interval()
        return 3600  # Default 1 hour 