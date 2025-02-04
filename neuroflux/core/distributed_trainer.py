import os
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.pipeline.sync import Pipe
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
from typing import Dict, Optional
from contextlib import nullcontext

from neuroflux.core.trainers import UnifiedTrainer
from neuroflux.core.unified_layer import UnifiedNeuroFlux
from neuroflux.monitoring.monitoring import PerformanceMonitor
from neuroflux.core.hypernetwork import UnifiedHyperAction, UnifiedMetrics

class DistributedNeuroFluxTrainer(UnifiedTrainer):
    """Extends UnifiedTrainer with distributed capabilities"""
    def __init__(
        self,
        model: UnifiedNeuroFlux,
        config: Dict,
        zero_stage: int = 3,
        use_fsdp: bool = True,
        pipeline_parallel: bool = True,
        grad_accum_steps: int = 1
    ):
        # Initialize base trainer first
        super().__init__(model, config)
        
        self.grad_accum_steps = grad_accum_steps
        self.current_micro_step = 0
        
        # Setup distributed environment
        self.setup_distributed()
        
        # Wrap model with distributed strategies
        self.model = self.wrap_distributed_model(
            self.model,
            zero_stage,
            use_fsdp,
            pipeline_parallel
        )
        
    def setup_distributed(self):
        """Initialize distributed training environment"""
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)
        
    def wrap_distributed_model(
        self,
        model: UnifiedNeuroFlux,
        zero_stage: int,
        use_fsdp: bool,
        pipeline_parallel: bool
    ) -> nn.Module:
        """Add distributed wrappers while preserving UnifiedNeuroFlux architecture"""
        model = model.cuda()
        
        # Apply gradient checkpointing
        model = checkpoint_wrapper.CheckpointWrapper(model)
        
        if use_fsdp:
            # FSDP with mixed precision
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.local_rank
            )
            
        elif pipeline_parallel:
            # Pipeline parallelism preserving unified architecture
            model = Pipe(model, chunks=8)
            
        # DeepSpeed ZeRO integration
        ds_config = {
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_accumulation_steps": self.grad_accum_steps,
            "train_micro_batch_size_per_gpu": self.config.batch_size // self.grad_accum_steps
        }
        
        model, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        return model
        
    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute training step with proper gradient accumulation"""
        # Scale batch for gradient accumulation
        micro_batch_size = batch.size(0) // self.grad_accum_steps
        micro_batch = batch[self.current_micro_step * micro_batch_size:
                          (self.current_micro_step + 1) * micro_batch_size]
        
        # Forward and backward pass with scaled loss
        with self.model.no_sync() if self.current_micro_step != self.grad_accum_steps - 1 else nullcontext():
            output = super().training_step(micro_batch)
            loss = output['loss'] / self.grad_accum_steps
            loss.backward()
        
        # Update on final micro step
        if self.current_micro_step == self.grad_accum_steps - 1:
            if hasattr(self.model, 'clip_grad_norm_'):
                self.model.clip_grad_norm_(self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        # Update micro step counter
        self.current_micro_step = (self.current_micro_step + 1) % self.grad_accum_steps
        
        return {
            'loss': loss.item() * self.grad_accum_steps,  # Return unscaled loss
            'step': self.current_step,
            'metrics': output.get('metrics', {})
        } 