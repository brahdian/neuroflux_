import torch
import psutil
import GPUtil
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import wandb
import json
import os
from datetime import datetime
from pathlib import Path
from .config_registry import ConfigRegistry
from .utils import ThreadSafeDict, CircularBuffer, PriorityQueue, AlertHandlerRegistry
from queue import Empty
from ..core.distributed_trainer import DistributedNeuroFluxTrainer as DistributedTrainer
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

MONITORED_METRICS = [
    'gpu_util', 'gpu_temp', 'gpu_memory', 'cpu_util', 'memory_util',
    'disk_usage', 'loss', 'gradient_norm', 'learning_rate', 'throughput',
    'expert_utilization', 'expert_load_balance', 'routing_entropy',
    'raid_health', 'recovery_time', 'parity_check_time', 'network_latency',
    'bandwidth_usage', 'sync_time'
]

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring"""
    sampling_rate: float = 1.0  # Hz
    history_size: int = 3600    # 1 hour of history
    alert_cooldown: int = 300   # 5 minutes between alerts
    gpu_temp_threshold: float = 80.0  # Celsius
    gpu_memory_threshold: float = 0.95  # 95% usage
    cpu_threshold: float = 0.90  # 90% usage
    memory_threshold: float = 0.85  # 85% usage
    throughput_drop_threshold: float = 0.3  # 30% drop
    log_dir: str = "logs"
    
    # New thresholds for enhanced monitoring
    gradient_norm_threshold: float = 100.0
    loss_spike_threshold: float = 5.0
    expert_imbalance_threshold: float = 0.3
    raid_recovery_time_threshold: float = 30.0  # seconds
    network_latency_threshold: float = 1.0  # seconds
    disk_usage_threshold: float = 0.90  # 90% usage
    
    # Initialize metrics history with default factory
    metrics_history: Dict[str, deque] = field(default_factory=lambda: {
        'gpu_util': deque(maxlen=3600),
        'gpu_temp': deque(maxlen=3600),
        'gpu_memory': deque(maxlen=3600),
        'cpu_util': deque(maxlen=3600),
        'memory_util': deque(maxlen=3600),
        'disk_usage': deque(maxlen=3600),
        'loss': deque(maxlen=3600),
        'gradient_norm': deque(maxlen=3600),
        'learning_rate': deque(maxlen=3600),
        'throughput': deque(maxlen=3600),
        'expert_utilization': deque(maxlen=3600),
        'expert_load_balance': deque(maxlen=3600),
        'routing_entropy': deque(maxlen=3600),
        'raid_health': deque(maxlen=3600),
        'recovery_time': deque(maxlen=3600),
        'parity_check_time': deque(maxlen=3600),
        'network_latency': deque(maxlen=3600),
        'bandwidth_usage': deque(maxlen=3600),
        'sync_time': deque(maxlen=3600)
    })

class PerformanceMonitor:
    """Thread-safe performance monitoring with resource management"""
    def __init__(self):
        self.config = ConfigRegistry.get_config()
        self._setup_logging()
        
        # Thread-safe metrics storage using RWLock
        self._metrics_lock = threading.RLock()
        self._metrics_history = ThreadSafeDict()
        
        # Resource limits
        self.max_memory_usage = 0.8 * psutil.virtual_memory().total
        self.max_history_size = self.config.history_size
        
        # Alert management with priorities
        self.alert_priorities = {
            'CRITICAL': 0,
            'ERROR': 1, 
            'WARNING': 2
        }
        self.alert_queue = PriorityQueue()
        self.alert_handlers = AlertHandlerRegistry()
        
        # Add cleanup configuration
        self.cleanup_interval = 3600  # Cleanup every hour
        self.last_cleanup = time.time()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
    def _initialize_monitoring(self):
        """Initialize monitoring with resource limits"""
        self.metrics_history = {
            metric: CircularBuffer(
                maxlen=self.max_history_size,
                dtype=np.float32  # Memory efficient storage
            )
            for metric in MONITORED_METRICS
        }
        
        # Start monitoring threads
        self.running = True
        self._start_monitoring()
        
    def _start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Resource-aware monitoring loop"""
        while self.running:
            try:
                # Check resource limits
                if self._check_resource_limits():
                    metrics = self._collect_metrics()
                    
                    with self._metrics_lock:
                        self._update_metrics(metrics)
                        
                    # Check anomalies
                    self._check_anomalies(metrics, time.time())
                    
                time.sleep(1.0 / self.config.sampling_rate)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                self._handle_monitoring_error(e)
                
    def _check_resource_limits(self) -> bool:
        """Check if within resource limits"""
        memory_usage = psutil.Process().memory_info().rss
        
        if memory_usage > self.max_memory_usage:
            self._handle_resource_limit("Memory usage exceeded")
            return False
            
        return True
        
    def _update_metrics(self, metrics: Dict[str, float]):
        """Thread-safe metrics update"""
        try:
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                
            # Cleanup old metrics if needed
            self._cleanup_old_metrics()
                
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")
            
    def _process_alerts(self):
        """Process alerts with retries and priorities"""
        while self.running:
            try:
                priority, alert = self.alert_queue.get(timeout=1.0)
                
                # Try each handler with retries
                for handler in self.alert_handlers.get_handlers(priority):
                    for attempt in range(3):  # 3 retries
                        try:
                            handler.send_alert(alert)
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.logger.error(
                                    f"Alert handler failed: {e}"
                                )
                            
            except Empty:
                continue
                
    def cleanup(self):
        """Proper resource cleanup"""
        self.running = False
        
        # Stop threads
        self.monitor_thread.join(timeout=5.0)
        self.alert_thread.join(timeout=5.0)
        
        # Cleanup resources
        with self._metrics_lock:
            self.metrics_history.clear()
            
        # Close handlers
        self.alert_handlers.cleanup()
        
        self.logger.info("Monitoring shutdown complete")

    def _initialize_baselines(self):
        """Initialize performance baselines for anomaly detection"""
        # System baselines
        self.baselines.update({
            'gpu_util_mean': 0.0,
            'memory_util_mean': 0.0,
            'throughput_mean': 0.0,
            'loss_mean': 0.0,
            'loss_std': 1.0
        })
        
    def _check_anomalies(self, metrics: Dict[str, float], current_time: float):
        """
        Enhanced anomaly detection with multiple alert levels
        
        Alert Levels:
        - WARNING: Potential issues that need attention
        - ERROR: Serious problems requiring immediate action
        - CRITICAL: System stability at risk
        """
        # System resource anomalies
        if metrics.get('gpu_util', 0) > self.config.gpu_threshold:
            self._alert('gpu_util', 'CRITICAL', f"GPU utilization critical: {metrics['gpu_util']:.1f}%", current_time)
            
        if metrics.get('memory_util', 0) > self.config.memory_threshold:
            self._alert('memory', 'ERROR', f"Memory usage high: {metrics['memory_util']:.1f}%", current_time)
            
        # Training anomalies
        if metrics.get('gradient_norm', 0) > self.config.gradient_norm_threshold:
            self._alert('gradient', 'WARNING', f"Gradient norm spike: {metrics['gradient_norm']:.1f}", current_time)
            
        loss = metrics.get('loss', 0)
        if abs(loss - self.baselines['loss_mean']) > self.config.loss_spike_threshold * self.baselines['loss_std']:
            self._alert('loss', 'ERROR', f"Unusual loss value: {loss:.3f}", current_time)
            
        # Expert utilization anomalies
        if metrics.get('expert_imbalance', 0) > self.config.expert_imbalance_threshold:
            self._alert('expert_balance', 'WARNING', f"Expert load imbalance: {metrics['expert_imbalance']:.2f}", current_time)
            
        # RAID system anomalies
        if metrics.get('raid_recovery_time', 0) > self.config.raid_recovery_time_threshold:
            self._alert('raid', 'CRITICAL', f"Slow RAID recovery: {metrics['raid_recovery_time']:.1f}s", current_time)
            
        # Network anomalies
        if metrics.get('network_latency', 0) > self.config.network_latency_threshold:
            self._alert('network', 'ERROR', f"High network latency: {metrics['network_latency']:.3f}s", current_time)

    def _alert(self, metric: str, level: str, message: str, current_time: float):
        """
        Enhanced alerting system with multiple channels and severity levels
        """
        if current_time - self.last_alert_time[metric] > self.config.alert_cooldown:
            self.last_alert_time[metric] = current_time
            
            # Format alert message
            alert_msg = f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
            
            # Send to all configured channels
            for channel, sender in self.alert_channels.items():
                try:
                    sender(level, alert_msg)
                except Exception as e:
                    self.logger.error(f"Failed to send alert to {channel}: {e}")

    def get_detailed_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive metrics summary with statistical analysis
        """
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                values_array = np.array(list(values))
                summary[metric] = {
                    'current': values_array[-1],
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'p95': np.percentile(values_array, 95),
                    'trend': self._compute_trend(values_array)
                }
                
        return summary

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend direction and magnitude using linear regression"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    @property
    def trainer(self):
        """Lazy load trainer to avoid circular imports"""
        if self._trainer is None:
            from ..core.distributed_trainer import DistributedNeuroFluxTrainer as DistributedTrainer
            self._trainer = DistributedTrainer
        return self._trainer
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('NeuroFluxMonitor')
        self.logger.setLevel(logging.INFO)
        
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_path = os.path.join(
            self.config.log_dir,
            f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        metrics = {}
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_util'] = np.mean([gpu.load for gpu in gpus])
                metrics['gpu_temp'] = np.mean([gpu.temperature for gpu in gpus])
                metrics['gpu_memory'] = np.mean([gpu.memoryUtil for gpu in gpus])
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # CPU metrics
        metrics['cpu_util'] = psutil.cpu_percent() / 100.0
        metrics['memory_util'] = psutil.virtual_memory().percent / 100.0
        
        return metrics
    
    def update_training_metrics(
        self,
        loss: float,
        throughput: float,
        step: int
    ):
        """Update training-specific metrics"""
        self.metrics_history['loss'].append(loss)
        self.metrics_history['throughput'].append(throughput)
        
        # Update baselines periodically
        if step % 100 == 0:
            self._update_baselines()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'loss': loss,
                'throughput': throughput,
                'step': step
            })
    
    def _update_baselines(self):
        """Update performance baselines"""
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                self.baselines[metric] = np.mean(list(values))
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log current metrics"""
        if self.use_wandb:
            wandb.log(metrics)
        
        # Save detailed metrics periodically
        if time.time() % 300 < 1.0:  # Every 5 minutes
            self._save_detailed_metrics()
    
    def _save_detailed_metrics(self):
        """Save detailed metrics to file"""
        detailed_metrics = {
            metric: list(values)
            for metric, values in self.metrics_history.items()
        }
        
        metrics_path = os.path.join(
            self.config.log_dir,
            f'metrics_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(detailed_metrics, f)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                values_list = list(values)
                summary[metric] = {
                    'mean': np.mean(values_list),
                    'std': np.std(values_list),
                    'min': np.min(values_list),
                    'max': np.max(values_list),
                    'last': values_list[-1]
                }
        
        return summary
    
    def get_throttling_recommendation(self) -> Optional[Dict[str, float]]:
        """Get throttling recommendations based on system state"""
        summary = self.get_summary()
        
        if 'gpu_util' in summary and summary['gpu_util']['mean'] > 0.95:
            return {
                'batch_size_factor': 0.8,
                'gradient_accumulation_steps': 2
            }
        
        if 'memory_util' in summary and summary['memory_util']['mean'] > 0.9:
            return {
                'batch_size_factor': 0.7,
                'activation_checkpointing': True
            }
        
        return None
    
    def _handle_resource_limit(self, message: str):
        """Handle resource limit exceeded"""
        self.logger.error(message)
        # Implement resource limit handling logic here

    def _handle_monitoring_error(self, e: Exception):
        """Handle monitoring error"""
        self.logger.error(f"Monitoring error: {e}")
        # Implement error handling logic here

    def _cleanup_old_metrics(self):
        """Periodic cleanup of old metrics"""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        with self._metrics_lock:
            for metric in list(self._metrics_history.keys()):
                # Remove metrics older than max history
                cutoff_time = current_time - self.max_history_size
                while (len(self._metrics_history[metric]) > 0 and 
                       self._metrics_history[metric][0][0] < cutoff_time):
                    self._metrics_history[metric].popleft()
                    
                # Remove empty metrics
                if len(self._metrics_history[metric]) == 0:
                    del self._metrics_history[metric]
                    
        self.last_cleanup = current_time
        
        # Log memory usage
        process = psutil.Process()
        self.logger.info(f"Memory usage after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    def _update_metrics(self, metrics: Dict[str, float]):
        """Thread-safe metrics update"""
        try:
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                
            # Cleanup old metrics if needed
            self._cleanup_old_metrics()
                
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")

    def _check_resource_limits(self) -> bool:
        """Check if within resource limits"""
        memory_usage = psutil.Process().memory_info().rss
        
        if memory_usage > self.max_memory_usage:
            self._handle_resource_limit("Memory usage exceeded")
            return False
            
        return True

    def _process_alerts(self):
        """Process alerts with retries and priorities"""
        while self.running:
            try:
                priority, alert = self.alert_queue.get(timeout=1.0)
                
                # Try each handler with retries
                for handler in self.alert_handlers.get_handlers(priority):
                    for attempt in range(3):  # 3 retries
                        try:
                            handler.send_alert(alert)
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.logger.error(
                                    f"Alert handler failed: {e}"
                                )
                            
            except Empty:
                continue
                
    def cleanup(self):
        """Proper resource cleanup"""
        self.running = False
        
        # Stop threads
        self.monitor_thread.join(timeout=5.0)
        self.alert_thread.join(timeout=5.0)
        
        # Cleanup resources
        with self._metrics_lock:
            self.metrics_history.clear()
            
        # Close handlers
        self.alert_handlers.cleanup()
        
        self.logger.info("Monitoring shutdown complete") 