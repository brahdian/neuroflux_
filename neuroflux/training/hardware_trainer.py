"""
NeuroFlux Hardware-Agnostic Trainer
==================================
Automatically selects and optimizes for available hardware (TPU/GPU)
"""

import torch
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class HardwareManager:
    """Manages hardware-specific setup and operations"""
    def __init__(self):
        self.device_type = self._detect_hardware()
        self.device = self._setup_device()
        
    def _detect_hardware(self) -> str:
        """Detect available hardware"""
        try:
            import torch_xla.core.xla_model as xm
            return "TPU"
        except ImportError:
            if torch.cuda.is_available():
                return "GPU"
            return "CPU"
            
    def _setup_device(self):
        """Setup appropriate device"""
        if self.device_type == "TPU":
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        elif self.device_type == "GPU":
            return torch.device("cuda")
        return torch.device("cpu")

class HardwareAwareTrainer(UnifiedTrainer):
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
        self.hw = HardwareManager()
        self.grad_accum = grad_accum
        
        # Hardware-specific setup
        self._setup_hardware_specific()
        
    def _setup_hardware_specific(self):
        """Configure hardware-specific optimizations"""
        self.use_mp = self.config.get('mixed_precision', True)
        
        # Setup optimizer first
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        if self.hw.device_type == "TPU":
            self._setup_tpu()
        elif self.hw.device_type == "GPU":
            self._setup_gpu()
        else:
            self._setup_cpu()
            
    def _setup_tpu(self):
        """TPU-specific setup"""
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        
        self.model = self.model.to(self.hw.device)
        
        # TPU-specific optimizations
        torch_xla._XLAC._xla_set_default_device(self.hw.device)
        self.loader = pl.MpDeviceLoader(self.train_loader, self.hw.device)
        xm.set_rng_state(None)
        
    def _setup_gpu(self):
        """GPU-specific setup"""
        if self.use_mp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.model = self.model.to(self.hw.device)
        
    def _setup_cpu(self):
        """CPU-specific setup"""
        self.use_mp = False  # Disable mixed precision for CPU
        self.model = self.model.to(self.hw.device)
        
    def _tpu_training_step(self, batch: torch.Tensor, is_final_step: bool) -> Tuple[float, Dict]:
        """TPU-specific training step"""
        import torch_xla.core.xla_model as xm
        
        with xm.GradientAccumulation(self.grad_accum):
            with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                loss, metrics = self._forward_pass(batch)
                scaled_loss = loss / self.grad_accum
                scaled_loss.backward()
                
                if is_final_step:
                    xm.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    xm.mark_step()
        
        metrics = {k: v.cpu().item() if torch.is_tensor(v) else v 
                  for k, v in metrics.items()}
        return loss.item(), metrics

    def _gpu_training_step(self, batch: torch.Tensor, is_final_step: bool) -> Tuple[float, Dict]:
        """GPU-specific training step"""
        with torch.cuda.amp.autocast(enabled=self.use_mp):
            loss, metrics = self._forward_pass(batch)
            scaled_loss = loss / self.grad_accum
            
        if self.use_mp:
            self.scaler.scale(scaled_loss).backward()
            if is_final_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            scaled_loss.backward()
            if is_final_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
        metrics = {k: v.cpu().item() if torch.is_tensor(v) else v 
                  for k, v in metrics.items()}
        return loss.item(), metrics

    def _cpu_training_step(self, batch: torch.Tensor, is_final_step: bool) -> Tuple[float, Dict]:
        """CPU-specific training step"""
        loss, metrics = self._forward_pass(batch)
        scaled_loss = loss / self.grad_accum
        scaled_loss.backward()
        
        if is_final_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        metrics = {k: v.cpu().item() if torch.is_tensor(v) else v 
                  for k, v in metrics.items()}
        return loss.item(), metrics

    def training_step(self, batch: torch.Tensor) -> Tuple[float, Dict]:
        """Hardware-aware training step"""
        try:
            batch = batch.to(self.hw.device, non_blocking=True)
            is_final_step = (self.current_step + 1) % self.grad_accum == 0
            
            if self.hw.device_type == "TPU":
                return self._tpu_training_step(batch, is_final_step)
            elif self.hw.device_type == "GPU":
                return self._gpu_training_step(batch, is_final_step)
            else:
                return self._cpu_training_step(batch, is_final_step)
                
        except Exception as e:
            logger.error(f"Training error on {self.hw.device_type}: {str(e)}")
            raise 

    def _save_checkpoint(self):
        """Hardware-aware checkpoint saving"""
        if (self.hw.device_type == "TPU" and not xm.is_master_ordinal()) or \
           (self.hw.device_type == "GPU" and not torch.distributed.get_rank() == 0):
            return
        
        # Move state dict to CPU for saving
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        checkpoint = {
            'model_state': cpu_state_dict,
            'step': self.current_step,
            'optimizer': self.optimizer.state_dict()
        }
        
        save_path = f"{self.checkpoint_dir}/model_{self.current_step}.pt"
        torch.save(checkpoint, save_path)
        
        if self.hw.device_type == "TPU":
            xm.mark_step()