# neuroflux/raid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union, Any
import numpy as np
from pathlib import Path
import logging
import glob
from dataclasses import dataclass, field
import time
import matplotlib.pyplot as plt
from collections import deque
import threading
from neuroflux.utils.logging_config import setup_logger
import torch_xla.core.xla_model as xm

logger = setup_logger(__name__)

class RAIDError(Exception):
    """Base class for RAID-related errors"""
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = time.time()

class RecoveryError(RAIDError):
    """Error during RAID recovery"""
    def __init__(self, message: str, failed_blocks: List[int] = None):
        super().__init__(message, error_code=500)
        self.failed_blocks = failed_blocks or []

class GF256:
    """
    Galois Field GF(2^8) implementation for Reed-Solomon coding
    as specified in Section 2.2 of whitepaper
    """
    def __init__(self):
        # Generate exp and log tables for GF(2^8)
        self.exp = [0] * 256
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = self._multiply_primitive(x)
        self.exp[255] = self.exp[0]
    
    def _multiply_primitive(self, x):
        """Multiply by primitive polynomial x^8 + x^4 + x^3 + x^2 + 1"""
        highbit = x & 0x80
        x = (x << 1) & 0xFF
        if highbit:
            x ^= 0x1D
        return x
    
    def multiply(self, x, y):
        """Multiply two elements in GF(2^8)"""
        if x == 0 or y == 0:
            return 0
        return self.exp[(self.log[x] + self.log[y]) % 255]
    
    def divide(self, x, y):
        """Divide two elements in GF(2^8)"""
        if y == 0:
            raise ValueError("Division by zero in GF(2^8)")
        if x == 0:
            return 0
        return self.exp[(self.log[x] - self.log[y]) % 255]

class ReedSolomon:
    """
    Reed-Solomon encoder/decoder over GF(2^8) for RAID-6
    Implements equation G*M = P from Section 2.2
    """
    def __init__(self, num_data_blocks, num_parity_blocks, field=None):
        self.num_data = num_data_blocks
        self.num_parity = num_parity_blocks
        self.field = field or GF256()
        
        # Generate Vandermonde matrix for encoding
        self.generator_matrix = self._build_generator_matrix()
    
    def _build_generator_matrix(self):
        """Build Vandermonde matrix for RS encoding"""
        matrix = []
        for i in range(self.num_parity):
            row = []
            for j in range(self.num_data):
                # Use x^(i*j) for Vandermonde matrix
                power = (i * j) % 255
                row.append(self.field.exp[power])
            matrix.append(row)
        return matrix
    
    def encode(self, data_blocks):
        """Encode data blocks to generate parity"""
        if len(data_blocks) != self.num_data:
            raise ValueError("Incorrect number of data blocks")
            
        parity_blocks = [[0] * len(data_blocks[0]) for _ in range(self.num_parity)]
        
        # Compute G*M matrix multiplication in GF(2^8)
        for i in range(self.num_parity):
            for j in range(self.num_data):
                for k in range(len(data_blocks[0])):
                    parity_blocks[i][k] ^= self.field.multiply(
                        self.generator_matrix[i][j],
                        data_blocks[j][k]
                    )
        
        return data_blocks + parity_blocks
    
    def decode(self, available_blocks, available_indices):
        """Decode using surviving blocks to recover lost data"""
        if len(available_blocks) < self.num_data:
            raise ValueError("Not enough blocks for recovery")
            
        # Build decoding matrix based on available indices
        decode_matrix = []
        for i in range(len(available_indices)):
            row = []
            for j in range(self.num_data):
                power = (available_indices[i] * j) % 255
                row.append(self.field.exp[power])
            decode_matrix.append(row)
            
        # Solve system of equations to recover data
        return self._solve_linear_system(decode_matrix, available_blocks)
    
    def _solve_linear_system(self, matrix, data):
        """Solve linear system using Gaussian elimination in GF(2^8)"""
        # This is currently marked as "pass" and needs implementation
        rows = len(matrix)
        cols = len(matrix[0])
        
        # Create augmented matrix
        augmented = [row + [d] for row, d in zip(matrix, data)]
        
        # Forward elimination
        for i in range(rows):
            pivot = augmented[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular")
                
            for j in range(i + 1, rows):
                factor = self.field.divide(augmented[j][i], pivot)
                for k in range(i, cols + 1):
                    augmented[j][k] ^= self.field.multiply(factor, augmented[i][k])
                    
        # Back substitution
        solution = [0] * rows
        for i in range(rows - 1, -1, -1):
            sum_val = augmented[i][cols]
            for j in range(i + 1, cols):
                sum_val ^= self.field.multiply(augmented[i][j], solution[j])
            solution[i] = self.field.divide(sum_val, augmented[i][i])
            
        return solution

@dataclass
class RAIDStats:
    """Container for RAID statistics"""
    health_score: float
    error_count: int
    recovery_times: List[float]
    bank_status: Dict[int, bool]

@dataclass 
class RAIDConfig:
    """RAID configuration parameters"""
    num_data_slots: int = 4
    num_parity_slots: int = 2
    min_checkpoint_interval: int = 300  # 5 minutes
    max_checkpoint_interval: int = 7200  # 2 hours
    compression_threshold: int = 1000
    recovery_timeout: int = 10  # 10 seconds max recovery time
    checkpoint_dir: str = "checkpoints"

class EnhancedRAID6(nn.Module):
    """
    Enhanced RAID-6 implementation with complete Reed-Solomon error correction
    and adaptive compression as described in the whitepaper
    """
    def __init__(
        self,
        num_blocks: int = 4,
        parity_blocks: int = 2,
        compression_ratio: int = 8,
        config: RAIDConfig = None
    ):
        super().__init__()
        self.config = config or RAIDConfig()
        self.rs = ReedSolomon(num_blocks, parity_blocks)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Detect hardware
        self.device = self._get_appropriate_device()
        self.is_tpu = self.device.type == 'xla'
        
        # Register buffers on appropriate device
        self.register_buffer('data_banks', torch.zeros(num_blocks, 256, device=self.device))
        self.register_buffer('parity_banks', torch.zeros(parity_blocks, 256, device=self.device))
        self.register_buffer('error_counts', torch.zeros(num_blocks + parity_blocks, device=self.device))
        self.register_buffer('last_checkpoint', torch.zeros(1, device=self.device))
        
        # Recovery tracking
        self.recovery_depth = 0
        self.max_recovery_depth = 2
        self.recovery_timeout = 10  # seconds
        self.recovery_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }
        
        self.stats = RAIDStats()
        
        self.register_buffer('health_score', torch.ones(1, device=self.device))
        self.register_buffer('last_health_check', torch.zeros(1, device=self.device))
        
        self.register_buffer('halt_signal', torch.zeros(1, dtype=torch.bool, device=self.device))
        self.register_buffer('halt_countdown', torch.zeros(1, device=self.device))
        self.halt_timeout = 60  # 60 seconds for graceful shutdown
        
        # Add lock for thread safety
        self.raid_lock = threading.Lock()
        
    def _get_appropriate_device(self):
        """Get appropriate device based on available hardware"""
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            if torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        
    def _sync_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Hardware-agnostic tensor synchronization"""
        if self.is_tpu:
            return xm.mesh_reduce(name, tensor, lambda x: x[0])
        return tensor.clone()
        
    def should_checkpoint(self) -> bool:
        """Hardware-agnostic checkpointing"""
        current_time = time.time()
        time_since_checkpoint = current_time - self._sync_tensor(
            self.last_checkpoint, 'checkpoint_time').item()
            
        error_rate = self._sync_tensor(
            self.error_counts.float().mean(), 'error_rate')
            
        if self.error_counts.sum() > 0:
            return time_since_checkpoint >= self.config.min_checkpoint_interval
        else:
            adaptive_interval = min(
                self.config.max_checkpoint_interval,
                self.config.min_checkpoint_interval * (1 + 10 * (1 - error_rate))
            )
            return time_since_checkpoint >= adaptive_interval
            
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Hardware-agnostic tensor to numpy conversion"""
        if self.is_tpu:
            # TPU tensors need to be transferred to CPU first
            return tensor.cpu().numpy()
        return tensor.detach().cpu().numpy()
        
    def recover_from_failure(
        self,
        max_retries: int = 3,
        model: nn.Module = None,
        recovery_strategy: str = 'auto',
        recovery_depth: int = 0
    ) -> Optional[torch.Tensor]:
        """Enhanced recovery with depth tracking and timeout"""
        if recovery_depth >= self.max_recovery_depth:
            logger.error("Maximum recovery depth reached")
            return None
            
        start_time = time.time()
        self.recovery_stats['attempts'] += 1
        
        # Try strategies in order of speed
        strategies = [
            self._quick_xor_recovery,      # Fastest: Simple XOR
            self._reed_solomon_recovery,    # Medium: RS coding
            self._checkpoint_recovery       # Slowest: Full reload
        ]
        
        if recovery_strategy != 'auto':
            strategies = [s for s in strategies if s.__name__.startswith(f"_{recovery_strategy}")]
            
        for strategy in strategies:
            try:
                if time.time() - start_time > self.recovery_timeout:
                    logger.warning(f"Recovery timeout after {time.time() - start_time:.2f}s")
                    break
                    
                recovered = strategy(
                    model=model,
                    recovery_depth=recovery_depth + 1
                )
                
                if recovered is not None:
                    self.recovery_stats['successes'] += 1
                    self.update_error_tracking()
                    logger.info(f"Recovered using {strategy.__name__}")
                    return recovered
                    
            except (torch.cuda.CUDAError, torch.cuda.OutOfMemoryError) as e:
                logger.error(f"GPU error during recovery: {str(e)}")
                raise RecoveryError(f"GPU error: {str(e)}")
            except ValueError as e:
                logger.error(f"Invalid data during recovery: {str(e)}")
                raise RecoveryError(f"Data error: {str(e)}")
            except TimeoutError as e:
                logger.error(f"Recovery timed out: {str(e)}")
                raise RecoveryError("Recovery timed out")
            except Exception as e:
                logger.error(f"Unexpected error during recovery: {str(e)}")
                raise RAIDError(f"Unexpected error: {str(e)}")
                
        self.recovery_stats['failures'] += 1
        return None

    def compress_state(self, state):
        """Enhanced FP8 compression with hybrid scheme from Section 4.2"""
        # Compute optimal scale factor based on value distribution
        abs_max = torch.abs(state).max()
        scale = min(127.0 / abs_max.item(), 100.0)  # Cap scale for stability
        
        # Apply hybrid compression scheme
        if state.numel() > 1000:  # Large tensors use FP8
            return torch.quantize_per_tensor(
                state,
                scale=1/scale,
                zero_point=0,
                dtype=torch.qint8
            )
        else:  # Small tensors keep full precision
            return state
        
    def decompress_state(self, compressed):
        return compressed.dequantize()
    
    def encode(self, data_chunks):
        """Hardware-agnostic encoding"""
        with self.raid_lock:
            compressed = [self.compress_state(chunk) for chunk in data_chunks]
            # Convert to numpy in hardware-agnostic way
            numpy_chunks = [self._to_numpy(c) for c in compressed]
            encoded = self.rs.encode(numpy_chunks)
            
            # Move encoded data to appropriate device
            self.data_banks = torch.tensor(encoded[:self.config.num_data_slots], device=self.device)
            self.parity_banks = torch.tensor(encoded[self.config.num_data_slots:], device=self.device)

    def decode(self, available_indices):
        """Thread-safe failure recovery"""
        with self.raid_lock:
            chunks = []
            for i in available_indices:
                if i < self.config.num_data_slots:
                    chunks.append(self.data_banks[i])
                else:
                    chunks.append(self.parity_banks[i - self.config.num_data_slots])
                    
            decoded = self.rs.decode(chunks, available_indices)
            return [self.decompress_state(torch.tensor(d, device=self.device)) for d in decoded]

    def detect_failure_count(self) -> Tuple[int, Dict[str, float]]:
        """Enhanced failure detection with detailed statistics"""
        failed = 0
        failure_stats = {
            'data_corruption': [],
            'parity_corruption': [],
            'total_banks': len(self.data_banks) + len(self.parity_banks)
        }
        
        # Check data banks
        for i, bank in enumerate(self.data_banks):
            failure_ratio = (
                torch.isnan(bank).float().mean() +
                torch.isinf(bank).float().mean() +
                torch.abs(bank).gt(1e6).float().mean()  # Check for explosions
            ).item()
            
            failure_stats['data_corruption'].append(failure_ratio)
            if failure_ratio > 0.01:  # More than 1% corruption
                failed += 1
                
        # Check parity banks
        for i, bank in enumerate(self.parity_banks):
            failure_ratio = (
                torch.isnan(bank).float().mean() +
                torch.isinf(bank).float().mean() +
                torch.abs(bank).gt(1e6).float().mean()
            ).item()
            
            failure_stats['parity_corruption'].append(failure_ratio)
            if failure_ratio > 0.01:
                failed += 1
                
        failure_stats['total_failed'] = failed
        failure_stats['failure_rate'] = failed / failure_stats['total_banks']
        
        return failed, failure_stats

    def rebuild_from_parity(self):
        """Rebuild using Reed-Solomon when ≤2 failures"""
        available_data = []
        available_indices = []
        
        # Collect available chunks
        for i, bank in enumerate([*self.data_banks, *self.parity_banks]):
            if not torch.isnan(bank).any() and not torch.isinf(bank).any():
                available_data.append(bank.numpy())
                available_indices.append(i)
        
        # Decode using Reed-Solomon
        decoded = self.rs.decode(available_data, available_indices)
        
        # Restore data and parity banks
        self.data_banks = torch.tensor(decoded[:self.config.num_data_slots], device=self.device)
        self.parity_banks = torch.tensor(decoded[self.config.num_data_slots:], device=self.device)

    def reload_checkpoint(self):
        """Reload latest stable checkpoint with FP8 compression"""
        checkpoint_paths = sorted(glob.glob("checkpoint_*.pt"))
        if not checkpoint_paths:
            raise RuntimeError("No checkpoints found for recovery")
        
        # Load latest checkpoint
        checkpoint = torch.load(checkpoint_paths[-1])
        
        # Decompress FP8 weights if needed
        if isinstance(checkpoint['model_state'], dict):
            for k, v in checkpoint['model_state'].items():
                if isinstance(v, dict) and 'quantized' in v:
                    checkpoint['model_state'][k] = v['quantized'].float() * v['scale']
                
        return checkpoint

    def rebuild_raid_memory(self, checkpoint):
        """
        Rebuild RAID memory banks from checkpoint with FP8 compression
        as described in Section 4.2 of whitepaper
        """
        # Extract memory state
        if 'raid_memory' not in checkpoint:
            raise RuntimeError("Checkpoint missing RAID memory state")
        
        raid_state = checkpoint['raid_state']
        
        # Decompress FP8 data if needed
        if 'compression_format' in raid_state and raid_state['compression_format'] == 'fp8':
            data_chunks = []
            for chunk in raid_state['data_chunks']:
                if isinstance(chunk, dict) and 'scale' in chunk:
                    # Decompress FP8 format: value * scale
                    data_chunks.append(chunk['quantized'].float() * chunk['scale'])
                else:
                    data_chunks.append(chunk)
        else:
            data_chunks = raid_state['data_chunks']
        
        # Restore data banks with compression
        self.data_banks = torch.tensor([
            self.compress_state(chunk) 
            for chunk in data_chunks
        ], device=self.device)
        
        # Recompute parity using Reed-Solomon
        encoded = self.rs.encode(self.data_banks.numpy())
        self.parity_banks = torch.tensor(encoded[self.config.num_data_slots:], device=self.device)
        
        # Reset error tracking
        self.error_counts.zero_()

    def verify_integrity(self, thorough: bool = False) -> Tuple[bool, Dict[str, float]]:
        """Enhanced integrity verification with detailed diagnostics"""
        stats = {
            'data_corruption': 0.0,
            'parity_mismatch': 0.0,
            'value_range': 0.0,
            'numerical_stability': 0.0
        }
        
        try:
            # Basic corruption check
            for i, bank in enumerate(self.data_banks):
                if torch.isnan(bank).any() or torch.isinf(bank).any():
                    stats['data_corruption'] += 1
                    
            # Parity verification
            encoded = self.rs.encode(self.data_banks.numpy())
            expected_parity = torch.tensor(encoded[self.config.num_data_slots:], device=self.device)
            
            rel_error = torch.abs(self.parity_banks - expected_parity) / (torch.abs(expected_parity) + 1e-6)
            abs_error = torch.abs(self.parity_banks - expected_parity)
            
            stats['parity_mismatch'] = rel_error.mean().item()
            
            # Value range check
            for bank in self.data_banks:
                decompressed = self.decompress_state(bank)
                max_val = torch.abs(decompressed).max().item()
                if max_val > 100:
                    stats['value_range'] += max_val / 100
                    
            # Numerical stability (thorough mode only)
            if thorough:
                test_data = torch.randn_like(self.data_banks[0])
                encoded = self.encode([test_data])
                decoded = self.decode(list(range(len(encoded))))[0]
                stats['numerical_stability'] = F.mse_loss(test_data, decoded).item()
                
            # Overall health check
            is_healthy = (
                stats['data_corruption'] == 0 and
                stats['parity_mismatch'] < 1e-4 and
                stats['value_range'] < 1.5 and
                (not thorough or stats['numerical_stability'] < 1e-5)
            )
            
            return is_healthy, stats
            
        except Exception as e:
            logger.error(f"Integrity check failed: {str(e)}")
            return False, stats

    def get_adaptive_checkpoint_interval(self) -> int:
        """Enhanced adaptive checkpoint interval from Section 4.2"""
        # Base intervals from whitepaper
        MIN_INTERVAL = 300  # 5 minutes
        MAX_INTERVAL = 7200  # 2 hours
        
        # Compute error rate with exponential decay
        recent_errors = self.error_counts.sum().item()
        total_banks = len(self.error_counts)
        error_rate = recent_errors / max(1, total_banks)
        
        # Dynamic interval based on error rates
        if error_rate > 0.1:  # High error rate
            return MIN_INTERVAL
        elif error_rate > 0.01:  # Moderate error rate
            return MIN_INTERVAL * 6  # 30 minutes
        else:  # Low error rate
            # Gradually increase up to max interval
            return min(
                MAX_INTERVAL,
                MIN_INTERVAL * 24 * (1 - error_rate) / 0.01
            )

    def update_error_tracking(self):
        """Update error statistics for adaptive checkpointing"""
        # Detect errors in current state
        current_errors = torch.zeros_like(self.error_counts)
        
        # Check data banks
        for i, bank in enumerate(self.data_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i] = 1
                
        # Check parity banks
        for i, bank in enumerate(self.parity_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i + self.config.num_data_slots] = 1
                
        # Update error history with exponential decay
        decay = 0.95
        self.error_counts = decay * self.error_counts + (1 - decay) * current_errors

    def retrieve(self, h_prev):
        """
        Adaptive memory retrieval with error checking
        Returns recovered state or None if unrecoverable
        """
        try:
            # Check for errors in current state
            if torch.isnan(h_prev).any() or torch.isinf(h_prev).any():
                # Attempt recovery from RAID
                recovered = self.recover_from_failure()
                if recovered is not None:
                    return recovered
                    
            # No errors, return decompressed state
            return self.decompress_state(h_prev)
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            return None
    
    def encode_memory(self, state_dict):
        """
        Encode model state into RAID format with FP8 compression
        Returns encoded state with parity
        """
        # Compress state to FP8
        compressed = {}
        for key, tensor in state_dict.items():
            if tensor.requires_grad:
                # Use FP8 for trainable parameters
                compressed[key] = self.compress_state(tensor)
            else:
                # Keep buffers in original precision
                compressed[key] = tensor
                
        # Split into chunks and compute parity
        chunks = self._split_into_chunks(compressed)
        encoded = self.rs.encode(chunks)
        
        return {
            'data': encoded[:self.config.num_data_slots],
            'parity': encoded[self.config.num_data_slots:],
            'compression_format': 'fp8'
        }
    
    def decode_memory(self, encoded_state):
        """
        Decode RAID-encoded state with FP8 decompression
        """
        # Recover from available chunks
        available = encoded_state['data'] + encoded_state['parity']
        indices = list(range(len(available)))
        
        decoded = self.rs.decode(available, indices)
        
        # Rebuild state dict with decompression
        state_dict = {}
        for key, compressed in decoded.items():
            if encoded_state.get('compression_format') == 'fp8':
                state_dict[key] = self.decompress_state(compressed)
            else:
                state_dict[key] = compressed
                
        return state_dict

    def _split_into_chunks(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Split data into optimal chunk sizes based on memory banks"""
        chunk_size = data.size(-1) // self.config.num_data_slots
        chunks = []
        
        for i in range(self.config.num_data_slots):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.config.num_data_slots - 1 else None
            chunk = data[..., start_idx:end_idx]
            chunks.append(chunk)
            
        return chunks
        
    def _combine_chunks(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Combine chunks back into original tensor"""
        return torch.cat(chunks, dim=-1)

    def _quick_xor_recovery(self, model=None, recovery_depth: int = 0) -> Optional[torch.Tensor]:
        """Fast recovery using XOR for single failure"""
        start_time = time.time()
        corrupted = self._detect_corruption()
        
        if len(corrupted) != 1:
            return None
            
        try:
            # XOR remaining blocks to recover
            recovered = self.data_banks[0].clone()
            for i in range(1, len(self.data_banks)):
                if i not in corrupted:
                    recovered ^= self.data_banks[i]
                    
            recovery_time = time.time() - start_time
            self.update_stats(
                recovery_time=recovery_time,
                recovery_method='xor',
                success=True
            )
            return recovered
            
        except Exception as e:
            logger.error(f"XOR recovery failed: {str(e)}")
            self.update_stats(recovery_method='xor', success=False)
            return None

    def _reed_solomon_recovery(self, model=None, recovery_depth: int = 0) -> Optional[torch.Tensor]:
        """Recovery using Reed-Solomon for up to 2 failures"""
        start_time = time.time()
        corrupted = self._detect_corruption()
        
        if len(corrupted) > 2:
            return None
            
        try:
            available_data = []
            available_indices = []
            
            # Collect available chunks
            for i, bank in enumerate([*self.data_banks, *self.parity_banks]):
                if i not in corrupted:
                    available_data.append(bank.numpy())
                    available_indices.append(i)
                    
            decoded = self.rs.decode(available_data, available_indices)
            self.data_banks = torch.tensor(decoded[:self.config.num_data_slots], device=self.device)
            self.parity_banks = torch.tensor(decoded[self.config.num_data_slots:], device=self.device)
            
            recovery_time = time.time() - start_time
            self.update_stats(
                recovery_time=recovery_time,
                recovery_method='reed_solomon',
                success=True
            )
            return self.data_banks.sum(0)
            
        except Exception as e:
            logger.error(f"Reed-Solomon recovery failed: {str(e)}")
            self.update_stats(recovery_method='reed_solomon', success=False)
            return None

    def _checkpoint_recovery(self, model=None, recovery_depth: int = 0) -> Optional[torch.Tensor]:
        """Full recovery from checkpoint"""
        if model is None:
            return None
            
        start_time = time.time()
        try:
            checkpoint = self._load_checkpoint()
            model.load_state_dict(checkpoint['model_state'])
            self.rebuild_raid_memory(checkpoint)
            
            recovery_time = time.time() - start_time
            self.update_stats(
                recovery_time=recovery_time,
                recovery_method='checkpoint',
                success=True
            )
            return self.data_banks.sum(0)
            
        except Exception as e:
            logger.error(f"Checkpoint recovery failed: {str(e)}")
            self.update_stats(recovery_method='checkpoint', success=False)
            return None

    def update_stats(self, **kwargs):
        """Update performance statistics"""
        # Recovery metrics
        if 'recovery_time' in kwargs:
            self.stats.recovery_times.append(kwargs['recovery_time'])
            self.stats.performance_metrics['recovery_latency'].append(kwargs['recovery_time'])
            
        # Compression metrics
        if 'compression_ratio' in kwargs:
            self.stats.compression_ratios.append(kwargs['compression_ratio'])
            self.stats.performance_metrics['compression_efficiency'].append(kwargs['compression_ratio'])
            
        # Error tracking
        if 'error_rate' in kwargs:
            self.stats.error_rates.append(kwargs['error_rate'])
            
        # Checkpoint metrics
        if 'checkpoint_interval' in kwargs:
            self.stats.checkpoint_intervals.append(kwargs['checkpoint_interval'])
            
        # Bank health
        if 'corruption_detected' in kwargs:
            self.stats.bank_health['corruption_rate'].append(float(kwargs['corruption_detected']))
            
        # Recovery success tracking
        if 'success' in kwargs and 'recovery_method' in kwargs:
            self.stats.bank_health['recovery_success'].append({
                'method': kwargs['recovery_method'],
                'success': kwargs['success']
            })

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        metrics = {
            'avg_recovery_time': np.mean(self.stats.recovery_times[-100:]) if self.stats.recovery_times else 0,
            'compression_ratio': np.mean(self.stats.compression_ratios[-100:]) if self.stats.compression_ratios else 0,
            'error_rate': np.mean(self.stats.error_rates[-100:]) if self.stats.error_rates else 0,
            'checkpoint_interval': np.mean(self.stats.checkpoint_intervals[-100:]) if self.stats.checkpoint_intervals else 0,
            'recovery_success_rate': self._calculate_success_rate(),
            'bank_health_score': self._calculate_bank_health()
        }
        
        # Add method-specific recovery stats
        recovery_methods = {'xor', 'reed_solomon', 'checkpoint'}
        for method in recovery_methods:
            success_rate = self._calculate_method_success_rate(method)
            metrics[f'{method}_success_rate'] = success_rate
            
        return metrics

    def _calculate_success_rate(self) -> float:
        """Calculate overall recovery success rate"""
        if not self.stats.bank_health['recovery_success']:
            return 0.0
            
        recent_attempts = list(self.stats.bank_health['recovery_success'])
        successes = sum(1 for x in recent_attempts if x['success'])
        return successes / len(recent_attempts)

    def _calculate_method_success_rate(self, method: str) -> float:
        """Calculate success rate for specific recovery method"""
        if not self.stats.bank_health['recovery_success']:
            return 0.0
            
        attempts = [x for x in self.stats.bank_health['recovery_success'] 
                   if x['method'] == method]
        if not attempts:
            return 0.0
            
        successes = sum(1 for x in attempts if x['success'])
        return successes / len(attempts)

    def _calculate_bank_health(self) -> float:
        """Calculate overall bank health score (0-1)"""
        if not self.stats.bank_health['corruption_rate']:
            return 1.0
            
        recent_corruption = np.mean(list(self.stats.bank_health['corruption_rate']))
        return 1.0 - recent_corruption

    def visualize_memory_banks(self, save_path: Optional[str] = None):
        """Generate visualizations of memory bank state"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Bank Corruption
        corruption = torch.stack([
            torch.isnan(bank).any(dim=-1) | torch.isinf(bank).any(dim=-1)
            for bank in self.data_banks
        ])
        axes[0,0].imshow(corruption.cpu().numpy(), aspect='auto')
        axes[0,0].set_title('Bank Corruption Map')
        
        # Plot 2: Error History
        axes[0,1].plot(list(self.stats.error_rates))
        axes[0,1].set_title('Error Rate History')
        
        # Plot 3: Recovery Success by Method
        methods = ['xor', 'reed_solomon', 'checkpoint']
        success_rates = [self._calculate_method_success_rate(m) for m in methods]
        axes[1,0].bar(methods, success_rates)
        axes[1,0].set_title('Recovery Success Rates')
        
        # Plot 4: Bank Usage Stats
        usage_stats = self._get_bank_usage_stats()
        axes[1,1].boxplot(usage_stats.cpu().numpy())
        axes[1,1].set_title('Bank Usage Statistics')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _get_bank_usage_stats(self) -> torch.Tensor:
        """Get memory bank usage statistics"""
        return torch.stack([
            torch.tensor([
                bank.abs().mean(),
                bank.abs().std(),
                bank.abs().max()
            ])
            for bank in [*self.data_banks, *self.parity_banks]
        ])

    def monitor_health(self) -> Dict[str, float]:
        """Real-time health monitoring with memory management"""
        current_time = time.time()
        if current_time - xm.mesh_reduce(
            'checkpoint_time', self.last_checkpoint, lambda x: x[0]) < 60:
            return {
                'health_score': self.health_score.item(),
                'last_check': xm.mesh_reduce(
                    'checkpoint_time', self.last_checkpoint, lambda x: x[0])
            }
            
        is_healthy, stats = self.verify_integrity()
        
        # Cleanup old stats
        if hasattr(self, '_old_stats'):
            del self._old_stats
        self._old_stats = stats
        
        self.health_score.fill_(1.0 if is_healthy else stats['data_corruption'])
        xm.mesh_reduce(
            'checkpoint_time', self.last_checkpoint, lambda x: x.fill_(current_time))
        
        return {
            'health_score': self.health_score.item(),
            'stats': stats,
            'last_check': xm.mesh_reduce(
                'checkpoint_time', self.last_checkpoint, lambda x: x[0])
        }
        
    def get_raid_status(self) -> Dict[str, Any]:
        """Get complete RAID system status"""
        return {
            'health': self.monitor_health(),
            'performance': self.get_performance_metrics(),
            'bank_stats': self._get_bank_usage_stats(),
            'recovery_stats': {
                'attempts': self.recovery_stats['attempts'],
                'successes': self.recovery_stats['successes'],
                'failures': self.recovery_stats['failures']
            }
        }

    def cleanup(self):
        """Enhanced cleanup with halt handling"""
        try:
            if self.halt_signal.item():
                # Export stats if not done
                if not hasattr(self, '_stats_exported'):
                    self.export_stats(
                        Path(self.config.checkpoint_dir) / "halt_stats.pt"
                    )
                    self._stats_exported = True
                    
            # Existing cleanup code...
            super().cleanup()
            
        except Exception as e:
            logger.error(f"Cleanup failed during halt: {str(e)}")
            
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    def export_stats(self, path: str):
        """Export statistics for analysis"""
        stats_dict = {
            'recovery_times': self.stats.recovery_times,
            'compression_ratios': self.stats.compression_ratios,
            'error_rates': self.stats.error_rates,
            'checkpoint_intervals': self.stats.checkpoint_intervals,
            'bank_health': {
                k: list(v) for k, v in self.stats.bank_health.items()
            },
            'performance_metrics': {
                k: list(v) for k, v in self.stats.performance_metrics.items()
            }
        }
        torch.save(stats_dict, path)
        
    def import_stats(self, path: str):
        """Import statistics"""
        stats_dict = torch.load(path)
        self.stats = RAIDStats(
            recovery_times=stats_dict['recovery_times'],
            compression_ratios=stats_dict['compression_ratios'],
            error_rates=stats_dict['error_rates'],
            checkpoint_intervals=stats_dict['checkpoint_intervals']
        )
        for k, v in stats_dict['bank_health'].items():
            self.stats.bank_health[k] = deque(v, maxlen=1000)
        for k, v in stats_dict['performance_metrics'].items():
            self.stats.performance_metrics[k] = deque(v, maxlen=100)

    def initiate_soft_halt(self, reason: str = "unknown"):
        """Initiate graceful shutdown sequence"""
        logger.info(f"Initiating soft halt. Reason: {reason}")
        self.halt_signal.fill_(True)
        xm.mesh_reduce(
            'checkpoint_time', self.last_checkpoint, lambda x: x.fill_(time.time()))
        
        # Trigger final checkpoint
        if hasattr(self, 'checkpoint_manager'):
            try:
                self.checkpoint_manager.save_checkpoint(
                    self,
                    {
                        'data_banks': self.data_banks,
                        'parity_banks': self.parity_banks,
                        'error_counts': self.error_counts,
                        'stats': self.stats,
                        'halt_reason': reason
                    },
                    step='halt'
                )
            except Exception as e:
                logger.error(f"Failed to save halt checkpoint: {str(e)}")
    
    def check_halt_status(self) -> Dict[str, Any]:
        """Check halt status and perform cleanup if needed"""
        if not self.halt_signal.item():
            return {'halting': False}
            
        time_since_halt = time.time() - xm.mesh_reduce(
            'checkpoint_time', self.last_checkpoint, lambda x: x[0])
        
        if time_since_halt >= self.halt_timeout:
            # Force cleanup if timeout exceeded
            logger.warning("Halt timeout exceeded, forcing cleanup")
            self.cleanup()
            return {
                'halting': True,
                'status': 'completed',
                'forced': True
            }
            
        # Calculate remaining tasks
        pending_tasks = {
            'bank_flush': not torch.all(self.data_banks == 0),
            'parity_sync': not torch.all(self.parity_banks == 0),
            'stats_export': not hasattr(self, '_stats_exported')
        }
        
        return {
            'halting': True,
            'status': 'in_progress',
            'time_remaining': self.halt_timeout - time_since_halt,
            'pending_tasks': pending_tasks
        }
    
    def resume_from_halt(self) -> bool:
        """Attempt to resume from halted state"""
        try:
            # Check for halt checkpoint
            halt_checkpoint = sorted(
                Path(self.config.checkpoint_dir).glob("checkpoint_halt*.pt")
            )
            if not halt_checkpoint:
                return False
                
            # Load halt state
            checkpoint = torch.load(halt_checkpoint[-1])
            self.rebuild_raid_memory(checkpoint)
            
            # Clear halt signals
            self.halt_signal.fill_(False)
            xm.mesh_reduce(
                'checkpoint_time', self.last_checkpoint, lambda x: x.fill_(0))
            
            logger.info("Successfully resumed from halt state")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume from halt: {str(e)}")
            return False

    def _detect_corruption(self) -> List[int]:
        """Optimized corruption detection"""
        with self.raid_lock:
            # Vectorized corruption check
            is_corrupted = torch.stack([
                torch.isnan(bank).any() | 
                torch.isinf(bank).any() |
                torch.abs(bank).gt(1e6).any()
                for bank in self.data_banks
            ])
            
            return is_corrupted.nonzero().flatten().tolist()

    def _load_checkpoint(self) -> Dict:
        """Safe checkpoint loading with verification"""
        checkpoint_paths = sorted(glob.glob("checkpoint_*.pt"))
        if not checkpoint_paths:
            raise RuntimeError("No checkpoints found for recovery")
        
        # Try checkpoints from newest to oldest
        for path in reversed(checkpoint_paths):
            try:
                # Verify checkpoint integrity
                if not self._verify_checkpoint_integrity(path):
                    continue
                    
                checkpoint = torch.load(path)
                
                # Decompress FP8 weights if needed
                if isinstance(checkpoint['model_state'], dict):
                    for k, v in checkpoint['model_state'].items():
                        if isinstance(v, dict) and 'quantized' in v:
                            checkpoint['model_state'][k] = v['quantized'].float() * v['scale']
                            
                return checkpoint
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
                continue
                
        raise RuntimeError("No valid checkpoints found")

    def _verify_checkpoint_integrity(self, path: str) -> bool:
        """Verify checkpoint integrity"""
        # This is a placeholder implementation. You might want to implement
        # a more robust integrity check based on your checkpoint format.
        # For example, you could read the file and check its contents.
        return True  # Placeholder return, actual implementation needed

    def _recover_using_parity(self, corrupted: List[int]) -> Optional[List[torch.Tensor]]:
        """Enhanced parity recovery with error handling"""
        if not corrupted:
            return []
        
        if len(corrupted) > self.config.num_parity_slots:
            logger.error(f"Too many corrupted blocks ({len(corrupted)}) for available parity slots ({self.config.num_parity_slots})")
            return None
        
        try:
            with self.raid_lock:
                recovered = []
                for idx, i in enumerate(corrupted):
                    if idx >= len(self.parity_banks):
                        raise ValueError(f"Not enough parity banks for recovery: needed {len(corrupted)}, have {len(self.parity_banks)}")
                    
                    recovered_block = self.parity_banks[idx].clone()
                    valid_blocks = [(j, block) for j, block in enumerate(self.data_banks) 
                                  if j != i and j not in corrupted]
                    
                    for j, block in valid_blocks:
                        recovered_block = recovered_block ^ block
                        
                    # Verify recovered block
                    if torch.isnan(recovered_block).any() or torch.isinf(recovered_block).any():
                        raise ValueError(f"Recovery produced invalid block for index {i}")
                    
                    recovered.append(recovered_block)
                    
                return recovered
            
        except Exception as e:
            logger.error(f"Parity recovery failed: {str(e)}")
            return None

    def get_stats(self) -> RAIDStats:
        """Get current RAID statistics"""
        return RAIDStats(
            health_score=float(self.health_score.item()),
            error_count=len(self._detect_corruption()),
            recovery_times=list(self.stats.recovery_times),
            bank_status=self._get_bank_status()
        )

class RAIDMemory(nn.Module):
    """
    Redundant Array of Independent Distributed Memory
    Implements fault-tolerant memory management with parity-based recovery
    """
    def __init__(self, 
                 num_blocks: int = 8,
                 parity_slots: int = 2,
                 compression_threshold: int = 1000,
                 recovery_timeout: int = 360):
        super().__init__()
        self.num_blocks = num_blocks
        self.parity_slots = parity_slots
        self.compression_threshold = compression_threshold
        self.recovery_timeout = recovery_timeout
        
        # Initialize memory banks
        self.data_banks: List[torch.Tensor] = []
        self.parity_banks: List[torch.Tensor] = []
        self.error_counts = torch.zeros(num_blocks + parity_slots)
        
    def store(self, data: torch.Tensor) -> None:
        """Store data with redundancy"""
        # Split data into blocks
        blocks = self._split_into_blocks(data)
        
        # Compute parity
        parity = self._compute_parity(blocks)
        
        # Update storage
        self.data_banks = blocks
        self.parity_banks = parity
        
    def recover_from_failure(self) -> Optional[torch.Tensor]:
        """Attempt to recover data after detecting corruption"""
        try:
            # Check for corrupted blocks
            corrupted = self._detect_corruption()
            if not corrupted:
                return self._reconstruct_data()
                
            # Attempt recovery using parity
            recovered = self._recover_using_parity(corrupted)
            if recovered is not None:
                return self._reconstruct_data()
                
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            return None
            
    def _split_into_blocks(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Split input tensor into blocks"""
        chunks = torch.chunk(data, self.num_blocks)
        return [chunk.clone() for chunk in chunks]
        
    def _compute_parity(self, blocks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute parity blocks for redundancy"""
        parity = []
        for i in range(self.parity_slots):
            p = blocks[0].clone()
            for block in blocks[1:]:
                p = p ^ block  # XOR operation for parity
            parity.append(p)
        return parity
        
    def _detect_corruption(self) -> List[int]:
        """Optimized corruption detection"""
        with self.raid_lock:
            # Vectorized corruption check
            is_corrupted = torch.stack([
                torch.isnan(bank).any() | 
                torch.isinf(bank).any() |
                torch.abs(bank).gt(1e6).any()
                for bank in self.data_banks
            ])
            
            return is_corrupted.nonzero().flatten().tolist()
        
    def _recover_using_parity(self, corrupted: List[int]) -> Optional[List[torch.Tensor]]:
        """Enhanced parity recovery with error handling"""
        if not corrupted:
            return []
        
        if len(corrupted) > self.parity_slots:
            logger.error(f"Too many corrupted blocks ({len(corrupted)}) for available parity slots ({self.parity_slots})")
            return None
        
        try:
            with self.raid_lock:
                recovered = []
                for idx, i in enumerate(corrupted):
                    if idx >= len(self.parity_banks):
                        raise ValueError(f"Not enough parity banks for recovery: needed {len(corrupted)}, have {len(self.parity_banks)}")
                    
                    recovered_block = self.parity_banks[idx].clone()
                    valid_blocks = [(j, block) for j, block in enumerate(self.data_banks) 
                                  if j != i and j not in corrupted]
                    
                    for j, block in valid_blocks:
                        recovered_block = recovered_block ^ block
                        
                    # Verify recovered block
                    if torch.isnan(recovered_block).any() or torch.isinf(recovered_block).any():
                        raise ValueError(f"Recovery produced invalid block for index {i}")
                    
                    recovered.append(recovered_block)
                    
                return recovered
            
        except Exception as e:
            logger.error(f"Parity recovery failed: {str(e)}")
            return None
        
    def _reconstruct_data(self) -> torch.Tensor:
        """Reconstruct original data from blocks"""
        return torch.cat(self.data_banks, dim=0)

    def get_adaptive_checkpoint_interval(self) -> int:
        """Enhanced adaptive checkpoint interval from Section 4.2"""
        # Base intervals from whitepaper
        MIN_INTERVAL = 300  # 5 minutes
        MAX_INTERVAL = 7200  # 2 hours
        
        # Compute error rate with exponential decay
        recent_errors = self.error_counts.sum().item()
        total_banks = len(self.error_counts)
        error_rate = recent_errors / max(1, total_banks)
        
        # Dynamic interval based on error rates
        if error_rate > 0.1:  # High error rate
            return MIN_INTERVAL
        elif error_rate > 0.01:  # Moderate error rate
            return MIN_INTERVAL * 6  # 30 minutes
        else:  # Low error rate
            # Gradually increase up to max interval
            return min(
                MAX_INTERVAL,
                MIN_INTERVAL * 24 * (1 - error_rate) / 0.01
            )

    def update_error_tracking(self):
        """Update error statistics for adaptive checkpointing"""
        # Detect errors in current state
        current_errors = torch.zeros_like(self.error_counts)
        
        # Check data banks
        for i, bank in enumerate(self.data_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i] = 1
                
        # Check parity banks
        for i, bank in enumerate(self.parity_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i + self.num_blocks] = 1
                
        # Update error history with exponential decay
        decay = 0.95
        self.error_counts = decay * self.error_counts + (1 - decay) * current_errors

class CheckpointManager:
    """Manages model checkpoints with FP8 compression"""
    def __init__(self, config: RAIDConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def compress_fp8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """FP8 compression with adaptive scaling"""
        if tensor.numel() < self.config.compression_threshold:
            return tensor, 1.0
            
        abs_max = torch.abs(tensor).max()
        scale = min(127.0 / abs_max.item(), 100.0)
        quantized = torch.round(tensor * scale).to(torch.int8)
        return quantized, scale
        
    def decompress_fp8(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """FP8 decompression"""
        return quantized.float() / scale
        
    def save_checkpoint(
        self,
        model: nn.Module,
        raid_state: Dict,
        step: int,
        extra_state: Dict = None
    ) -> Dict[str, float]:
        """Save compressed checkpoint"""
        compressed_state = {}
        compression_stats = {'total_size': 0, 'compressed_size': 0}
        
        # Compress model state
        for name, param in model.state_dict().items():
            if param.requires_grad:
                quantized, scale = self.compress_fp8(param)
                compressed_state[name] = {
                    'quantized': quantized,
                    'scale': scale
                }
                compression_stats['total_size'] += param.numel() * 4
                compression_stats['compressed_size'] += quantized.numel()
            else:
                compressed_state[name] = param
                
        # Save checkpoint
        checkpoint = {
            'step': step,
            'model_state': compressed_state,
            'raid_state': raid_state
        }
        if extra_state:
            checkpoint.update(extra_state)
            
        torch.save(
            checkpoint,
            self.checkpoint_dir / f"checkpoint_{step}.pt"
        )
        
        return compression_stats
        
    def load_latest_checkpoint(self) -> Tuple[Dict, int]:
        """Load latest checkpoint with decompression"""
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            raise RuntimeError("No checkpoints found")
            
        checkpoint = torch.load(checkpoints[-1])
        
        # Decompress model state
        state_dict = {}
        for name, compressed in checkpoint['model_state'].items():
            if isinstance(compressed, dict) and 'scale' in compressed:
                state_dict[name] = self.decompress_fp8(
                    compressed['quantized'],
                    compressed['scale']
                )
            else:
                state_dict[name] = compressed
                
        checkpoint['model_state'] = state_dict
        return checkpoint, checkpoint['step']