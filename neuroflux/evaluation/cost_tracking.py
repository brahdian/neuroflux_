"""
Cost Tracking System
===================
Tracks compute, storage and network costs during evaluation.
"""

import time
import psutil
import GPUtil
from typing import Dict, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks resource usage and associated costs"""
    
    def __init__(self, cost_config: Dict[str, float]):
        self.config = cost_config
        self.start_time = None
        self.total_cost = 0.0
        self.peak_memory = 0.0
        
        # Resource tracking
        self.gpu_hours = 0.0
        self.network_gb = 0.0
        self.storage_gb = 0.0
        
    @contextmanager
    def __call__(self):
        """Context manager for tracking costs during execution"""
        self.start_tracking()
        try:
            yield
        finally:
            self.stop_tracking()
            
    def start_tracking(self):
        """Start tracking resource usage"""
        self.start_time = time.time()
        
        # Get initial resource usage
        self.initial_gpu = self._get_gpu_usage()
        self.initial_network = self._get_network_usage()
        self.initial_storage = self._get_storage_usage()
        
    def stop_tracking(self):
        """Stop tracking and compute costs"""
        if not self.start_time:
            return
            
        duration = time.time() - self.start_time
        
        # Compute resource usage
        gpu_usage = self._get_gpu_usage() - self.initial_gpu
        network_usage = self._get_network_usage() - self.initial_network
        storage_usage = self._get_storage_usage() - self.initial_storage
        
        # Update totals
        self.gpu_hours += duration / 3600  # Convert to hours
        self.network_gb += network_usage / 1024**3  # Convert to GB
        self.storage_gb += storage_usage / 1024**3  # Convert to GB
        
        # Compute costs
        gpu_cost = self.gpu_hours * self.config["gpu_cost_per_hour"]
        network_cost = self.network_gb * self.config["network_cost_per_gb"]
        storage_cost = self.storage_gb * self.config["storage_cost_per_gb"]
        
        self.total_cost = gpu_cost + network_cost + storage_cost
        
        # Track peak memory
        self.peak_memory = max(self.peak_memory, self._get_gpu_memory())
        
    def _get_gpu_usage(self) -> float:
        """Get GPU utilization"""
        try:
            return GPUtil.getGPUs()[0].load * 100
        except:
            return 0.0
            
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in GB"""
        try:
            return GPUtil.getGPUs()[0].memoryUsed / 1024
        except:
            return 0.0
            
    def _get_network_usage(self) -> float:
        """Get network usage in bytes"""
        return psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        
    def _get_storage_usage(self) -> float:
        """Get storage usage in bytes"""
        return psutil.disk_usage('/').used
        
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown"""
        return {
            "gpu_cost": self.gpu_hours * self.config["gpu_cost_per_hour"],
            "network_cost": self.network_gb * self.config["network_cost_per_gb"],
            "storage_cost": self.storage_gb * self.config["storage_cost_per_gb"],
            "total_cost": self.total_cost
        } 