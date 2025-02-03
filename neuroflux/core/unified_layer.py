"""
NeuroFlux Unified Architecture
=============================
Combines SSM dynamics with unified parameter sharing and curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
from system.raid import ReedSolomon, GF256, EnhancedRAID6, RAIDConfig
import time
from core.hypernetwork import DifferentiableHyperNetwork, UnifiedHyperNetwork
import logging
logger = logging.getLogger(__name__)
@dataclass
class RoutingStats:
    """Statistics for GPRO routing"""
    mean_load: torch.Tensor
    load_std: torch.Tensor
    usage_count: torch.Tensor
    importance: torch.Tensor
    
    @classmethod
    def init_stats(cls, num_experts: int, device: torch.device):
        return cls(
            mean_load=torch.zeros(1, device=device),
            load_std=torch.ones(1, device=device),
            usage_count=torch.zeros(num_experts, device=device),
            importance=torch.ones(num_experts, device=device)
        )

@dataclass
class RAIDState:
    """RAID memory state container"""
    data_banks: List[torch.Tensor]
    parity_banks: List[torch.Tensor]
    error_counts: torch.Tensor
    last_update: int = 0

@dataclass
class UnifiedState:
    """Unified state container for all components"""
    ssm: torch.Tensor  # [batch_size, hidden_size]
    memory: List[torch.Tensor]  # List[[batch_size, hidden_size]]
    kl_stats: Dict[str, torch.Tensor] = None
    expert_history: Optional[torch.Tensor] = None
    curriculum_level: int = 0
    value_stats: Dict[str, torch.Tensor] = None  # Track value statistics
    trust_region: Dict[str, torch.Tensor] = None  # KL trust region stats
    routing_stats: Optional[RoutingStats] = None
    raid_state: Optional[RAIDState] = None  # Enhanced RAID state tracking
    
    @classmethod
    def init_state(
        cls,
        batch_size: int,
        hidden_size: int,
        num_memory_cells: int,
        device: torch.device
    ) -> 'UnifiedState':
        return cls(
            ssm=torch.zeros(batch_size, hidden_size, device=device),
            memory=[
                torch.zeros(batch_size, hidden_size, device=device)
                for _ in range(num_memory_cells)
            ],
            kl_stats={'avg': torch.zeros(1, device=device), 
                     'std': torch.ones(1, device=device)},
            expert_history=None,
            curriculum_level=0
        )

    def update_value_stats(self, value: torch.Tensor, decay: float = 0.99):
        if self.value_stats is None:
            self.value_stats = {
                'mean': torch.zeros_like(value),
                'std': torch.ones_like(value)
            }
        self.value_stats['mean'] = (
            decay * self.value_stats['mean'] + 
            (1 - decay) * value.mean()
        )
        self.value_stats['std'] = (
            decay * self.value_stats['std'] + 
            (1 - decay) * value.std()
        )

class SSMKernel(nn.Module):
    """Dedicated SSM computation kernel"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(hidden_size, 2 * hidden_size)
        self.kernel_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize with proper scaling
        nn.init.normal_(self.kernel_proj.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        proj = self.in_proj(x)
        delta, gamma = proj.chunk(2, dim=-1)
        delta = F.softplus(delta)  # Ensure positive timesteps
        
        # Discretized state update
        kernel = self.kernel_proj(x)
        new_state = (
            state * torch.exp(-delta) + 
            kernel * (1 - torch.exp(-delta)) * gamma
        )
        return new_state

class RLOutput(NamedTuple):
    """Container for RL-specific outputs"""
    value: torch.Tensor
    policy_logits: torch.Tensor
    entropy: torch.Tensor

class TimeoutContext:
    """Context manager for timeout"""
    def __init__(self, seconds):
        self.seconds = seconds
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if time.time() - self.start_time > self.seconds:
            raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

class UnifiedNeuroFlux(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        num_memory_cells: int = 4,
        dropout: float = 0.1,
        init_temp: float = 1.0,
        trust_clip: float = 0.2,  # Trust region clipping
        rl_scale: float = 1.0,    # RL loss scaling
        # Add GPRO hyperparameters
        load_balance_decay: float = 0.99,
        importance_decay: float = 0.999,
        routing_temp: float = 0.1,
        min_routing_prob: float = 0.0001,
        # Add RAID parameters
        raid_slots: int = 4,
        parity_slots: int = 2,
        raid_update_freq: int = 100,
        compression_type: str = 'fp8',
        raid_config: Optional[RAIDConfig] = None,
        use_hypernetwork: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_memory_cells = num_memory_cells
        
        # Core components
        self.ssm = SSMKernel(hidden_size)
        
        # Unified parameter tensor with proper initialization
        self.unified_weights = nn.Parameter(
            torch.empty(hidden_size, hidden_size * 3)
        )
        nn.init.normal_(self.unified_weights, std=0.02 / math.sqrt(3 * hidden_size))
        
        # Memory cell timescales (learnable)
        self.memory_scales = nn.Parameter(
            torch.linspace(0, -2, num_memory_cells)
        )
        
        # Expert routing
        self.route_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Curriculum learning
        self.register_buffer('curriculum_progress', torch.zeros(1))
        self.register_buffer('temp', torch.full([1], init_temp))
        
        # Value head for RL
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Policy head for RL
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Trust region clipping
        self.trust_clip = trust_clip
        self.rl_scale = rl_scale
        
        # GPRO specific parameters
        self.load_balance_decay = load_balance_decay
        self.importance_decay = importance_decay
        self.routing_temp = routing_temp
        self.min_routing_prob = min_routing_prob
        
        # Expert importance projection
        self.importance_proj = nn.Linear(hidden_size, num_experts)
        
        # RAID configuration
        self.raid_slots = raid_slots
        self.parity_slots = parity_slots
        self.raid_update_freq = raid_update_freq
        self.register_buffer('steps_since_raid', torch.zeros(1))
        
        # Initialize enhanced RAID with config
        self.raid = EnhancedRAID6(
            num_blocks=raid_slots,
            parity_blocks=parity_slots,
            compression_ratio=8 if compression_type == 'fp8' else 1,
            config=raid_config
        )
        
        # Add hypernetwork for dynamic control
        self.use_hypernetwork = use_hypernetwork
        if use_hypernetwork:
            self.hypernet = UnifiedHyperNetwork(
                hidden_size=hidden_size,
                num_memory_cells=num_memory_cells,
                use_trust_region=True
            )
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[UnifiedState] = None,
        step: Optional[int] = None,
        return_attention: bool = False,
        failed_slots: List[int] = None
    ) -> Tuple[torch.Tensor, UnifiedState, Dict]:
        batch_size = x.size(0)
        
        if state is None:
            state = UnifiedState.init_state(
                batch_size, self.hidden_size,
                self.num_memory_cells, x.device
            )
            
        # Check halt status first
        if hasattr(self.raid, 'halt_signal') and self.raid.halt_signal.item():
            halt_status = self.raid.check_halt_status()
            if halt_status['status'] == 'completed':
                raise RuntimeError("System is halted")
            
        # Split unified parameters
        mem_w, route_w, value_w = self.unified_weights.chunk(3, dim=1)
        
        # 1. SSM-XLSTM fusion with RAID memory
        # Following whitepaper equation:
        # h_t = e^{Δ(W_A x̃_t)} h_{t-1} + Δ(W_B x̃_t) x_t + Σ α^(k) c_{t-1}^(k)
        # where x̃_t = x_t + RAID-Retrieve(h_{t-1})
        
        # Get RAID-enhanced input
        raid_memory = self.raid.retrieve(state.ssm) if state.raid_state is not None else None
        x_enhanced = x + raid_memory if raid_memory is not None else x
        
        # Get dynamic parameters
        if self.use_hypernetwork:
            hyper_params = self.hypernet(state.ssm, state.memory)
            
            # Apply SSM parameters
            ssm_state = self.ssm(
                x_enhanced, 
                state.ssm,
                delta=hyper_params['ssm']['delta'],
                gamma=hyper_params['ssm']['gamma']
            )
            
            # Apply MoE parameters
            routing_temp = hyper_params['moe']['temperature']
            balance_factor = hyper_params['moe']['balance_factor']
            
            # Apply memory gates
            memory_gates = F.linear(x_enhanced, mem_w)
            i, f, o = memory_gates.chunk(3, dim=-1)
            i, f, o = map(torch.sigmoid, (i, f, o))
            
            # Update each memory cell with its timescale
            memory_scales = torch.softmax(self.memory_scales, dim=0)
            new_memories = []
            for scale, prev_mem in zip(memory_scales, state.memory):
                cell = i * torch.tanh(F.linear(x_enhanced, mem_w)) + f * prev_mem
                cell = cell * scale
                new_memories.append(cell)
            
            # Combine memories with output gate
            memory_state = sum(new_memories) * o
        else:
            # Use default parameters
            ssm_state = self.ssm(x_enhanced, state.ssm)
            routing_temp = self.routing_temp
            balance_factor = 1.0
            
            # 2. Multi-scale memory integration
            memory_gates = F.linear(x_enhanced, mem_w)
            i, f, o = memory_gates.chunk(3, dim=-1)
            i, f, o = map(torch.sigmoid, (i, f, o))
            
            # Update each memory cell with its timescale
            memory_scales = torch.softmax(self.memory_scales, dim=0)
            new_memories = []
            for scale, prev_mem in zip(memory_scales, state.memory):
                cell = i * torch.tanh(F.linear(x_enhanced, mem_w)) + f * prev_mem
                cell = cell * scale
                new_memories.append(cell)
            
            # Combine memories with output gate
            memory_state = sum(new_memories) * o
        
        # 3. Expert routing with GPRO
        route_input = self.route_norm(ssm_state + memory_state)
        routing_logits = F.linear(route_input, route_w)
        
        # Compute routing probabilities
        k = min(2 + state.curriculum_level, self.num_experts)
        routing_probs, indices = self.compute_routing_probabilities(
            routing_logits, state, k
        )
        
        # Update routing statistics
        if self.training:
            self.update_routing_stats(
                state,
                indices,
                routing_probs,
                torch.sigmoid(self.importance_proj(route_input))
            )
        
        # 4. Value computation
        combined_state = ssm_state + memory_state
        expert_values = F.linear(combined_state, value_w)
        
        # Compute expert outputs
        expert_output = self._compute_expert_output(
            combined_state, indices, routing_probs
        )
        expert_output = self.dropout(expert_output)
        
        # Update state
        new_state = UnifiedState(
            ssm=ssm_state,
            memory=new_memories,
            kl_stats=state.kl_stats,
            expert_history=self._update_expert_history(
                state.expert_history, indices
            ),
            curriculum_level=state.curriculum_level,
            value_stats=state.value_stats,
            trust_region=state.trust_region,
            routing_stats=state.routing_stats,
            raid_state=state.raid_state
        )
        
        info = {
            'routing': {
                'probs': routing_probs,
                'indices': indices,
                'temp': routing_temp
            },
            'curriculum': {
                'level': state.curriculum_level,
                'progress': self.curriculum_progress
            }
        }
        
        if return_attention:
            info['attention'] = self._compute_attention_weights(
                combined_state, indices, routing_probs
            )
            
        # Add RL outputs
        rl_outputs = self.compute_rl_outputs(
            combined_state,
            routing_probs
        )
        
        info.update({
            'rl': {
                'value': rl_outputs.value,
                'policy_logits': rl_outputs.policy_logits,
                'entropy': rl_outputs.entropy
            }
        })
        
        # Update value statistics
        if state.value_stats is not None:
            state.update_value_stats(rl_outputs.value)
            
        # Update RAID state periodically or on failure detection
        if self.training:
            self.steps_since_raid += 1
            
        needs_raid_update = (
            self.steps_since_raid >= self.raid_update_freq or
            failed_slots is not None
        )
        
        if needs_raid_update:
            raid_state = self._update_raid_state(state)
            self.steps_since_raid.zero_()
        else:
            raid_state = state.raid_state

        # Handle checkpointing
        if self.training and step is not None and self.raid.should_checkpoint():
            compression_stats = self.raid.checkpoint_manager.save_checkpoint(
                self,
                {
                    'data_banks': state.raid_state.data_banks,
                    'parity_banks': state.raid_state.parity_banks,
                    'error_counts': state.raid_state.error_counts
                },
                step
            )
            self.raid.last_checkpoint.fill_(time.time())
        
        # Enhanced recovery
        if failed_slots is not None:
            recovered_state = self.raid.recover_from_failure(
                max_retries=3,
                model=self
            )
            if recovered_state is not None:
                x = x + recovered_state
            
        # Update state with new RAID information
        new_state = UnifiedState(
            ssm=ssm_state,
            memory=new_memories,
            kl_stats=state.kl_stats,
            expert_history=self._update_expert_history(
                state.expert_history, indices
            ),
            curriculum_level=state.curriculum_level,
            value_stats=state.value_stats,
            trust_region=state.trust_region,
            routing_stats=state.routing_stats,
            raid_state=raid_state
        )

        # Add RAID info to output
        info['raid'] = {
            'error_counts': self.raid.error_counts.clone(),
            'checkpoint_interval': self.raid.get_adaptive_checkpoint_interval(),
            'recovered': recovered_state is not None
        }
        
        # Add halt status to info dict
        if hasattr(self.raid, 'halt_signal'):
            info['halt_status'] = self.raid.check_halt_status()
        
        return expert_output, new_state, info
        
    def _compute_expert_output(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert outputs with unified parameters"""
        expert_indices = indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        expert_weights = F.linear(state, self.unified_weights)
        selected_weights = torch.gather(
            expert_weights.view(-1, self.num_experts, self.hidden_size),
            1,
            expert_indices
        )
        return torch.einsum('bke,bk->be', selected_weights, probs)
        
    def _update_expert_history(
        self,
        history: Optional[torch.Tensor],
        indices: torch.Tensor
    ) -> torch.Tensor:
        if history is None:
            history = torch.zeros(
                self.num_experts,
                device=indices.device
            )
        return history.scatter_add_(
            0,
            indices.view(-1),
            torch.ones_like(indices.view(-1), dtype=torch.float)
        )
        
    def update_curriculum(self, progress: float) -> None:
        """Update curriculum learning progress"""
        self.curriculum_progress.fill_(progress)

    def _compute_attention_weights(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights for attention mechanism"""
        expert_indices = indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        expert_weights = self.unified_weights.view(
            self.num_experts, self.hidden_size, -1
        )
        selected_weights = expert_weights.gather(0, expert_indices)
        
        attention_weights = torch.einsum('bke,bk->be', 
            selected_weights,
            probs
        )
        return attention_weights 

    def compute_rl_outputs(
        self, 
        features: torch.Tensor,
        routing_probs: torch.Tensor
    ) -> RLOutput:
        """Compute RL-specific outputs"""
        # Value prediction
        value = self.value_head(features).squeeze(-1)
        
        # Policy logits
        policy_logits = self.policy_head(features)
        
        # Policy entropy
        entropy = -(routing_probs * torch.log(routing_probs + 1e-8)).sum(-1)
        
        return RLOutput(value, policy_logits, entropy)

    def compute_routing_probabilities(
        self,
        logits: torch.Tensor,
        state: UnifiedState,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPRO routing with unified temperature control"""
        # Get unified temperature from hypernetwork
        if self.use_hypernetwork:
            hyper_params = self.hypernet(state.ssm, self.get_unified_metrics())
            routing_temp = hyper_params.temp  # Single temperature for unified control
        else:
            routing_temp = self.routing_temp
        
        # Apply temperature to unified computation
        if self.training:
            # Adjust logits based on expert usage (shared computation)
            usage_penalties = -torch.log1p(state.routing_stats.usage_count)
            adjusted_logits = logits + usage_penalties.unsqueeze(0)
            
            # Temperature-controlled exploration
            routing_scores = adjusted_logits / routing_temp
        else:
            routing_scores = logits / routing_temp
        
        # Top-k selection with importance weighting
        top_k_scores, indices = routing_scores.topk(k, dim=-1)
        
        # Temperature-scaled probabilities
        probs = F.softmax(top_k_scores, dim=-1)
        
        return probs, indices
        
    def update_routing_stats(
        self,
        state: UnifiedState,
        indices: torch.Tensor,
        probs: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> None:
        """Update GPRO routing statistics"""
        if state.routing_stats is None:
            return
            
        # Update expert usage counts
        batch_size = indices.size(0)
        expert_load = torch.zeros(
            self.num_experts,
            device=indices.device
        ).scatter_add_(
            0,
            indices.view(-1),
            probs.view(-1)
        ) / batch_size
        
        # Update running statistics
        state.routing_stats.mean_load = (
            self.load_balance_decay * state.routing_stats.mean_load +
            (1 - self.load_balance_decay) * expert_load.mean()
        )
        state.routing_stats.load_std = (
            self.load_balance_decay * state.routing_stats.load_std +
            (1 - self.load_balance_decay) * expert_load.std()
        )
        
        # Update usage counts
        state.routing_stats.usage_count = (
            self.importance_decay * state.routing_stats.usage_count +
            (1 - self.importance_decay) * expert_load
        )
        
        # Update importance scores
        state.routing_stats.importance = (
            self.importance_decay * state.routing_stats.importance +
            (1 - self.importance_decay) * importance_scores.mean(0)
        )

    def compute_gpro_loss(
        self,
        old_probs: torch.Tensor,
        new_probs: torch.Tensor,
        old_values: torch.Tensor,
        new_values: torch.Tensor,
        advantages: torch.Tensor,
        state: UnifiedState
    ) -> Dict[str, torch.Tensor]:
        """GPRO loss computation"""
        # Policy loss with importance sampling
        ratio = new_probs / (old_probs + 1e-8)
        policy_loss = -(ratio * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, old_values)
        
        # Load balancing loss
        if state.routing_stats is not None:
            load_std = state.routing_stats.load_std
            load_balance_loss = load_std / (state.routing_stats.mean_load + 1e-8)
        else:
            load_balance_loss = torch.tensor(0.0, device=new_probs.device)
            
        # Importance regularization
        importance_reg = -torch.mean(
            torch.log(state.routing_stats.importance + 1e-8)
        )
        
        # Combined loss
        total_loss = (
            self.rl_scale * policy_loss +
            0.5 * value_loss +
            0.1 * load_balance_loss +
            0.01 * importance_reg
        )
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'load_balance_loss': load_balance_loss,
            'importance_reg': importance_reg,
            'total_loss': total_loss
        } 

    def _update_raid_state(self, state: UnifiedState) -> Optional[RAIDState]:
        """Update RAID state with timeout protection"""
        try:
            with TimeoutContext(10):  # 10 second timeout
                combined_state = state.ssm + sum(state.memory)
                self.raid.encode([combined_state])
                
                return RAIDState(
                    data_banks=self.raid.data_banks.clone(),
                    parity_banks=self.raid.parity_banks.clone(),
                    error_counts=self.raid.error_counts.clone(),
                    last_update=int(self.steps_since_raid.item())
                )
        except TimeoutError:
            logger.error("RAID state update timed out")
            return None
        except Exception as e:
            logger.error(f"RAID state update failed: {str(e)}")
            return None

    def _recover_from_raid(
        self,
        state: UnifiedState,
        failed_slots: List[int]
    ) -> Optional[torch.Tensor]:
        """Recover state using Reed-Solomon decoding"""
        if state.raid_state is None:
            return None
            
        # Update RAID error tracking
        self.raid.error_counts = state.raid_state.error_counts
        self.raid.data_banks = state.raid_state.data_banks
        self.raid.parity_banks = state.raid_state.parity_banks
        
        try:
            # Attempt recovery using enhanced RAID-6
            recovered = self.raid.recover_from_failure()
            if recovered is not None:
                # Update error tracking
                self.raid.update_error_tracking()
                return recovered
        except Exception as e:
            print(f"RAID recovery failed: {str(e)}")
            return None 