# neuroflux/hypernetworks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, NamedTuple
import numpy as np
from torch.distributions import Normal, Independent
from dataclasses import dataclass
from torch.distributions import kl_divergence

class UnifiedHyperAction(NamedTuple):
    """Unified hyperparameter actions"""
    delta: torch.Tensor      # Time-scale for SSM and memory
    temp: torch.Tensor       # Temperature for routing
    scale: torch.Tensor      # Unified weight scaling

class UnifiedMetrics(NamedTuple):
    """Core unified metrics"""
    state_health: torch.Tensor     # Combined state stability
    param_efficiency: torch.Tensor # Parameter sharing effectiveness
    system_perf: torch.Tensor     # Overall system performance

class UnifiedHyperNetwork(nn.Module):
    """
    Simplified hypernetwork for UnifiedNeuroFlux's shared architecture.
    Single unified network controlling shared parameters.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Single unified network for all functionality
        self.unified_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Single parameter generator
        self.param_net = nn.Linear(hidden_size // 2, 9)  # means, stds, uncertainty
        
        # Single shared memory buffer
        self.register_buffer('shared_memory', torch.zeros(hidden_size // 2))
        
        # Parameter bounds
        self.register_buffer('param_bounds', torch.tensor([
            [0.1, 2.0],  # delta bounds
            [0.1, 1.0],  # temp bounds
            [0.5, 2.0]   # scale bounds
        ]))
        
    def forward(
        self,
        state: torch.Tensor,
        metrics: UnifiedMetrics
    ) -> UnifiedHyperAction:
        # Extract unified features
        features = self.unified_net(state)
        
        # Update shared memory
        self.shared_memory = 0.9 * self.shared_memory + 0.1 * features.mean(0)
        
        # Generate parameters and uncertainty
        outputs = self.param_net(features)
        means, log_stds, uncertainty = outputs.chunk(3, dim=-1)
        
        # Sample with uncertainty during training
        if self.training:
            log_stds = log_stds + uncertainty
            dist = Normal(means, log_stds.exp())
            actions = dist.rsample()
        else:
            actions = means
            
        # Scale to bounds
        actions = self._scale_to_bounds(actions)
        
        return UnifiedHyperAction(*actions.chunk(3, dim=-1))
        
    def _scale_to_bounds(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to parameter bounds"""
        min_bounds = self.param_bounds[:, 0]
        max_bounds = self.param_bounds[:, 1]
        return min_bounds + (max_bounds - min_bounds) * torch.sigmoid(actions)

class UncertaintyEstimator(nn.Module):
    """Enhanced uncertainty estimator with diversity"""
    def __init__(self, hidden_size: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)  # mean, log_var, confidence
        )
        
        # Diversity promoting loss weight
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, log_var, confidence = out.chunk(3, dim=-1)
        uncertainty = torch.exp(log_var) * torch.sigmoid(confidence)
        return mean, uncertainty

class UnifiedUncertaintyNet(nn.Module):
    """Uncertainty estimation for unified architecture"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3)  # unified uncertainty
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Single uncertainty for unified parameters
        return F.softplus(out)

class UnifiedParamNet(nn.Module):
    """Parameter generation for unified architecture"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 6)  # mean and std for 3 unified params
        )
        
    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor) -> Normal:
        out = self.net(x)
        mean, log_std = out.chunk(2, dim=-1)
        # Apply unified uncertainty
        log_std = log_std + uncertainty
        return Normal(mean, log_std.exp())

def init_hypernetwork(model: nn.Module) -> UnifiedHyperNetwork:
    """Initialize hypernetwork with model's hidden dimension"""
    hidden_size = model.config.hidden_size
    return UnifiedHyperNetwork(hidden_size)

class MetaController(nn.Module):
    """Meta-learning controller for hyperparameter adaptation"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Meta state tracking
        self.register_buffer('meta_state', torch.zeros(hidden_size))
        self.register_buffer('success_rate', torch.zeros(3))  # For each param type
        
        # Meta networks
        self.state_encoder = nn.GRUCell(hidden_size, hidden_size)
        self.adaptation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3)  # Learning rates per parameter type
        )
        
        # Performance tracking
        self.register_buffer('performance_window', torch.zeros(100))
        
    def update(self, features: torch.Tensor, metrics: UnifiedMetrics) -> torch.Tensor:
        """Update meta-state and compute adaptation rates"""
        # Update meta state
        self.meta_state = self.state_encoder(features, self.meta_state)
        
        # Update success rate based on metrics
        self.success_rate = 0.9 * self.success_rate + 0.1 * torch.tensor([
            metrics.state_health.mean(),
            metrics.param_efficiency.mean(),
            metrics.system_perf.mean()
        ])
        
        # Compute adaptation rates with success modulation
        base_rates = self.adaptation_net(self.meta_state)
        return base_rates * self.success_rate.unsqueeze(-1)

class EnhancedCurriculum:
    """Advanced curriculum with dynamic adjustment"""
    def __init__(self):
        self.phases = {
            'warmup': {
                'duration': 500,
                'targets': {
                    'sharing': 0.3,
                    'coherence': 0.5,
                    'complexity': 0.25
                }
            },
            'exploration': {
                'duration': 1000,
                'targets': {
                    'sharing': 0.5,
                    'coherence': 0.7,
                    'complexity': 0.5
                }
            },
            'exploitation': {
                'duration': 1500,
                'targets': {
                    'sharing': 0.7,
                    'coherence': 0.8,
                    'complexity': 0.75
                }
            },
            'refinement': {
                'duration': 2000,
                'targets': {
                    'sharing': 0.9,
                    'coherence': 0.9,
                    'complexity': 1.0
                }
            }
        }
        
        # Dynamic adjustment
        self.performance_threshold = 0.8
        self.adaptation_rate = 0.1
        
    def get_targets(self, step: int, performance: float) -> Dict[str, float]:
        """Get current targets with dynamic adjustment"""
        phase = self._get_current_phase(step)
        targets = self.phases[phase]['targets'].copy()
        
        # Adjust based on performance
        if performance > self.performance_threshold:
            # Accelerate curriculum
            for key in targets:
                targets[key] = min(1.0, targets[key] * (1 + self.adaptation_rate))
        elif performance < self.performance_threshold * 0.8:
            # Slow down curriculum
            for key in targets:
                targets[key] = max(0.1, targets[key] * (1 - self.adaptation_rate))
                
        return targets

class EnhancedUncertaintyEnsemble(nn.Module):
    """Advanced uncertainty estimation with diversity and calibration"""
    def __init__(self, hidden_size: int, ensemble_size: int = 5):
        super().__init__()
        
        # Ensemble members with diversity
        self.estimators = nn.ModuleList([
            UncertaintyEstimator(hidden_size, seed=i)
            for i in range(ensemble_size)
        ])
        
        # Calibration network
        self.calibration_net = nn.Sequential(
            nn.Linear(ensemble_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ensemble_size)
        )
        
        # Diversity loss weight
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
        # Calibration tracking
        self.register_buffer('calibration_scores', torch.ones(ensemble_size))
        self.register_buffer('prediction_history', torch.zeros(100, ensemble_size))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass with calibrated ensemble prediction"""
        ensemble_means = []
        ensemble_uncertainties = []
        
        # Get predictions from each estimator
        for estimator in self.estimators:
            mean, uncertainty = estimator(features)
            ensemble_means.append(mean)
            ensemble_uncertainties.append(uncertainty)
            
        # Stack predictions
        means = torch.stack(ensemble_means)
        uncertainties = torch.stack(ensemble_uncertainties)
        
        # Get calibration weights
        calibration_input = torch.cat([
            means.mean(0),
            uncertainties.mean(0)
        ], dim=-1)
        calibration_weights = F.softmax(
            self.calibration_net(calibration_input) * self.calibration_scores,
            dim=-1
        )
        
        # Compute calibrated uncertainty
        total_uncertainty = (
            (calibration_weights.unsqueeze(-1) * uncertainties).sum(0) +
            (calibration_weights.unsqueeze(-1) * (means - means.mean(0)).pow(2)).sum(0)
        ).sqrt()
        
        # Update calibration tracking
        self._update_calibration(means, uncertainties)
        
        return total_uncertainty
        
    def _update_calibration(self, means: torch.Tensor, uncertainties: torch.Tensor):
        """Update calibration scores based on prediction accuracy"""
        with torch.no_grad():
            # Update prediction history
            self.prediction_history = torch.roll(self.prediction_history, -1, dims=0)
            self.prediction_history[-1] = means.mean(-1)
            
            # Compute prediction error
            error = (means - means.mean(0)).abs().mean(-1)
            expected_error = uncertainties.mean(-1)
            
            # Update calibration scores based on error/uncertainty ratio
            calibration = torch.exp(-(error - expected_error).abs())
            self.calibration_scores = 0.9 * self.calibration_scores + 0.1 * calibration