import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Expert(nn.Module):
    """Expert module implementing the enhanced architecture"""
    def __init__(self, d_model: int, d_ff: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Multi-scale feature transformation
        self.input_proj = nn.Linear(d_model, d_ff)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ff1': nn.Linear(d_ff, d_ff * 2),
                'ff2': nn.Linear(d_ff * 2, d_ff),
                'norm': nn.LayerNorm(d_ff),
                'gate': nn.Linear(d_ff, 1, bias=False)
            }) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_ff, d_model)
        
        # Memory integration
        self.memory_key = nn.Linear(d_ff, d_model)
        self.memory_value = nn.Linear(d_ff, d_model)
        self.memory_query = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        
        # Adaptive layer processing
        for layer in self.layers:
            h2 = layer['ff1'](h)
            h2 = F.gelu(h2)
            h2 = layer['ff2'](h2)
            h2 = layer['norm'](h2)
            
            gate = torch.sigmoid(layer['gate'](h))
            h = h + self.dropout(gate * h2)
        
        # Memory integration
        if memory is not None:
            query = self.memory_query(x)
            key = self.memory_key(h)
            value = self.memory_value(h)
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = F.softmax(scores, dim=-1)
            memory_output = torch.matmul(attn, value)
            
            new_memory = torch.cat([memory[:, 1:], h[:, -1:]], dim=1)
            h = h + self.dropout(memory_output)
        else:
            new_memory = h[:, -1:]
        
        output = self.output_proj(h)
        output = self.norm(x + self.dropout(output))
        
        return output, new_memory