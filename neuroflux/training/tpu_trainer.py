"""
NeuroFlux TPU Training
=====================
TPU-optimized training implementation integrated with NeuroFlux framework
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

from ..core.trainers import UnifiedTrainer
from ..core.unified_layer import UnifiedNeuroFlux
from ..system.raid import EnhancedRAID6
from ..monitoring.monitoring import PerformanceMonitor
from ..core.tpu_utils import TPUManager

logger = logging.getLogger(__name__)

class TPUNeuroFluxTrainer(UnifiedTrainer):
    """TPU-optimized NeuroFlux trainer"""
    
    def __init__(
        self,
        model: UnifiedNeuroFlux,
        raid: EnhancedRAID6,
        monitor: PerformanceMonitor,
        config: Dict,
        checkpoint_dir: str = "./checkpoints",
        grad_accum: int = 1
    ):
        super().__init__(model, raid, monitor, config)
        self.grad_accum = grad_accum
        self.device = xm.xla_device()
        
        # TPU-specific mixed precision
        if self.use_mp:
            # TPU uses bfloat16 instead of float16
            self.model = self.model.to(torch.bfloat16)
            # No scaler needed for TPU bfloat16
        
        # Move model to TPU
        self.model = self.model.to(self.device)
        
        # TPU-specific optimizations
        self._setup_tpu_optimizations()
        
        # TPU-specific configs
        self.num_cores = 8  # Colab TPU v2-8
        self.batch_size = config.get('batch_size', 32)
        
        self.tpu_manager = TPUManager()
        self.raid = EnhancedRAID6(
            num_blocks=self.config.raid_blocks,
            parity_blocks=self.config.parity_blocks
        )
        
    def _setup_tpu_optimizations(self):
        """Configure TPU-specific optimizations"""
        # Enable dynamic batch shape handling
        torch_xla._XLAC._xla_set_default_device(self.device)
        
        # Set up TPU-optimized data loading
        self.loader = pl.MpDeviceLoader(self.train_loader, self.device)
        
        # Configure memory optimization
        xm.set_rng_state(None)
        
    def _setup_training(self):
        """Setup TPU training environment"""
        # Move model to TPU
        self.model = self.model.to(self.device)
        
        # Setup TPU optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Setup data parallel
        if xm.xrt_world_size() > 1:
            self.model = xmp.MpModelWrapper(self.model)
            
    def _setup_mixed_precision(self):
        """TPU-specific mixed precision setup"""
        if self.use_mp:
            # TPU uses bfloat16 instead of float16
            self.model = self.model.to(torch.bfloat16)
            self._mp_device = xm.xla_device(devkind='TPU')
        
    def training_step(self, batch: torch.Tensor) -> Tuple[float, Dict]:
        """Execute training step with TPU-specific handling"""
        try:
            # Ensure batch is on TPU
            batch = batch.to(self.device, non_blocking=True)
            
            # Gradient accumulation handling
            is_final_step = (self.current_step + 1) % self.grad_accum == 0
            
            # Forward pass with XLA optimization
            with xm.GradientAccumulation(self.grad_accum):
                with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                    loss, metrics = self._forward_pass(batch)
                    scaled_loss = loss / self.grad_accum
                    scaled_loss.backward()
                    
                    if is_final_step:
                        xm.optimizer_step(self.optimizer)
                        self.optimizer.zero_grad()
                        xm.mark_step()
                        
            # Ensure metrics are on CPU
            metrics = {k: v.cpu().item() if torch.is_tensor(v) else v 
                      for k, v in metrics.items()}
                    
            return loss.item(), metrics
            
        except Exception as e:
            logger.error(f"TPU training error: {str(e)}")
            xm.mark_step()  # Ensure XLA is in consistent state
            raise
        
    def _save_checkpoint(self):
        """Save TPU-compatible checkpoint"""
        if xm.is_master_ordinal():
            # Save on CPU to avoid TPU memory issues
            cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
            checkpoint = {
                'model_state': cpu_state_dict,
                'step': self.current_step,
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(checkpoint, 
                      f"{self.checkpoint_dir}/model_{self.current_step}.pt")
            xm.mark_step() 