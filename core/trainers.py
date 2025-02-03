"""
NeuroFlux Training Module
========================
Implements unified training protocol from whitepaper with three-phase curriculum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import math
import io
from dataclasses import dataclass
from typing import List, Dict
import time


from ..system.checkpoint import CheckpointManager
from ..system.raid import EnhancedRAID6
from ..core.hypernetwork import DifferentiableHyperNetwork
from .unified_layer import UnifiedNeuroFlux
import logging
logger = logging.getLogger(__name__)

@dataclass
class UsageStats:
    """Container for expert usage statistics"""
    total_calls: torch.Tensor
    recent_usage: Optional[torch.Tensor]
    peak_load: torch.Tensor
    avg_load: torch.Tensor
    compute_efficiency: torch.Tensor
    load_imbalance: float
    utilization: torch.Tensor

class ExpertUsageTracker:
    """Tracks and analyzes expert usage patterns over time"""
    def __init__(self, num_experts: int, history_size: int = 1000):
        self.num_experts = num_experts
        self.history_size = history_size
        
        # Usage statistics
        self.total_calls = torch.zeros(num_experts)
        self.recent_usage = deque(maxlen=history_size)
        self.usage_timestamps = deque(maxlen=history_size)
        
        # Load statistics
        self.peak_load = torch.zeros(num_experts)
        self.avg_load = torch.zeros(num_experts)
        
        # Efficiency metrics
        self.last_update = time.time()
        self.compute_efficiency = torch.zeros(num_experts)
        
    def update(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor) -> Dict:
        """Update usage statistics with new batch of expert assignments"""
        current_time = time.time()
        batch_size = expert_indices.size(0)
        
        # Compute current batch usage
        batch_usage = torch.bincount(
            expert_indices.view(-1),
            weights=expert_weights.view(-1),
            minlength=self.num_experts
        )
        
        # Update total usage
        self.total_calls += batch_usage
        
        # Update recent usage history
        self.recent_usage.append(batch_usage)
        self.usage_timestamps.append(current_time)
        
        # Update load statistics
        self.peak_load = torch.maximum(self.peak_load, batch_usage)
        self.avg_load = self._compute_moving_average(batch_usage)
        
        # Update efficiency metrics
        time_delta = current_time - self.last_update
        self.compute_efficiency = self._update_efficiency(batch_usage, time_delta)
        self.last_update = current_time
        
        return self._get_current_stats(batch_size)
        
    def _compute_moving_average(self, new_usage: torch.Tensor) -> torch.Tensor:
        """Compute exponential moving average of expert usage"""
        alpha = 0.99
        if not self.recent_usage:
            return new_usage
            
        current_avg = torch.stack(list(self.recent_usage)).mean(dim=0)
        return alpha * current_avg + (1 - alpha) * new_usage
        
    def _update_efficiency(self, batch_usage: torch.Tensor, time_delta: float) -> torch.Tensor:
        """Update compute efficiency metrics"""
        # Compute operations per second for each expert
        ops_per_second = batch_usage / max(time_delta, 1e-6)
        
        # Exponential moving average of efficiency
        alpha = 0.95
        return alpha * self.compute_efficiency + (1 - alpha) * ops_per_second
        
    def _get_current_stats(self, batch_size: int) -> Dict:
        """Generate current usage statistics"""
        return {
            'total_calls': self.total_calls,
            'recent_usage': torch.stack(list(self.recent_usage)) if self.recent_usage else None,
            'peak_load': self.peak_load,
            'avg_load': self.avg_load,
            'compute_efficiency': self.compute_efficiency,
            'load_imbalance': self._compute_load_imbalance(),
            'utilization': self._compute_utilization(batch_size)
        }
        
    def _compute_load_imbalance(self) -> float:
        """Compute load imbalance metric"""
        if not self.recent_usage:
            return 0.0
            
        recent_usage = torch.stack(list(self.recent_usage))
        mean_usage = recent_usage.mean(dim=0)
        std_usage = recent_usage.std(dim=0)
        return (std_usage / (mean_usage + 1e-6)).mean().item()
        
    def _compute_utilization(self, batch_size: int) -> torch.Tensor:
        """Compute expert utilization rates"""
        if not self.recent_usage:
            return torch.zeros(self.num_experts)
            
        recent_usage = torch.stack(list(self.recent_usage))
        total_possible = batch_size * len(self.recent_usage)
        return recent_usage.sum(dim=0) / total_possible
        
    def get_expert_rankings(self) -> List[int]:
        """Rank experts by their recent utilization"""
        if not self.recent_usage:
            return list(range(self.num_experts))
            
        recent_usage = torch.stack(list(self.recent_usage))
        mean_usage = recent_usage.mean(dim=0)
        return mean_usage.argsort(descending=True).tolist()
        
    def get_underutilized_experts(self, threshold: float = 0.1) -> List[int]:
        """Identify underutilized experts"""
        if not self.recent_usage:
            return []
            
        utilization = self._compute_utilization(1)  # batch_size=1 for relative utilization
        return (utilization < threshold).nonzero().squeeze(-1).tolist()
        
    def reset(self):
        """Reset all tracking statistics"""
        self.__init__(self.num_experts, self.history_size)

class MetricTracker:
    """Enhanced metric tracking with exponential moving averages"""
    def __init__(self, window_size=100, decay=0.99):
        self.window_size = window_size
        self.decay = decay
        self.values = deque(maxlen=window_size)
        self.ema = None
        self.best_value = float('inf')
        
    def update(self, value):
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.ema * self.decay + value * (1 - self.decay)
        if value < self.best_value:
            self.best_value = value
            
    def get_stats(self):
        return {
            'current': self.values[-1] if self.values else None,
            'ema': self.ema,
            'best': self.best_value,
            'avg': sum(self.values) / len(self.values) if self.values else None,
            'std': np.std(list(self.values)) if len(self.values) > 1 else 0
        }


class UnifiedTrainer:
    """Trainer for UnifiedNeuroFlux with true parameter sharing"""
    def __init__(self, model: UnifiedNeuroFlux, config: Dict):
        self.model = model
        
        # Single optimizer for unified parameters
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Training progress
        self.total_steps = config.total_steps
        self.current_step = 0
        
        # Unified metrics tracking
        self.metrics = MetricTracker()
        
        # Expert usage tracking
        self.expert_tracker = ExpertUsageTracker(
            num_experts=config.num_experts,
            history_size=1000
        )
        
        # RAID checkpointing
        self.checkpoint_manager = CheckpointManager(config.raid_config)
        
    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute unified training step"""
        # Forward pass with unified computation
        output, state, info = self.model(
            batch,
            step=self.current_step
        )
        
        # Update expert usage stats
        usage_stats = self.expert_tracker.update(
            info['routing']['indices'],
            info['routing']['weights']
        )
        
        # Compute unified loss (GPRO handles exploration/exploitation naturally)
        loss = self.model.compute_gpro_loss(
            state=state,
            info=info,
            usage_stats=usage_stats
        )
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track unified metrics
        self.metrics.update({
            'loss': loss.item(),
            'param_efficiency': state.metrics.param_efficiency,
            'system_perf': state.metrics.system_perf,
            'raid_health': state.metrics.raid_health,
            'expert_usage': usage_stats
        })
        
        # Handle RAID checkpointing
        if self.model.raid.should_checkpoint(self.current_step):
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                raid_state=state.raid_state,
                step=self.current_step
            )
        
        self.current_step += 1
        
        return {
            'loss': loss.item(),
            'step': self.current_step,
            'metrics': self.metrics.get_stats()
        }