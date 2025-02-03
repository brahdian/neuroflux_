"""
MTTR (Mean Time To Recovery) Tracking
===================================
Tracks system recovery times and reliability metrics.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class RecoveryEvent:
    """Container for recovery event data"""
    start_time: float
    end_time: float
    failure_type: str
    recovery_method: str
    success: bool
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class MTTRTracker:
    """Tracks system recovery times and reliability metrics"""
    
    def __init__(self, window_size: int = 100):
        self.events = deque(maxlen=window_size)
        self.current_recovery = None
        self.total_failures = 0
        self.successful_recoveries = 0
        
    def start_recovery(self, failure_type: str):
        """Record start of recovery attempt"""
        if self.current_recovery:
            logger.warning("Recovery already in progress")
            return
            
        self.current_recovery = {
            "start_time": time.time(),
            "failure_type": failure_type
        }
        self.total_failures += 1
        
    def end_recovery(self, recovery_method: str, success: bool):
        """Record end of recovery attempt"""
        if not self.current_recovery:
            logger.warning("No recovery in progress")
            return
            
        event = RecoveryEvent(
            start_time=self.current_recovery["start_time"],
            end_time=time.time(),
            failure_type=self.current_recovery["failure_type"],
            recovery_method=recovery_method,
            success=success
        )
        
        self.events.append(event)
        if success:
            self.successful_recoveries += 1
            
        self.current_recovery = None
        
    def get_mttr(self, window: Optional[int] = None) -> Optional[float]:
        """Calculate MTTR over specified window"""
        if not self.events:
            return None
            
        recent_events = list(self.events)[-window:] if window else list(self.events)
        successful_recoveries = [e for e in recent_events if e.success]
        
        if not successful_recoveries:
            return None
            
        return np.mean([e.duration for e in successful_recoveries])
        
    def get_recovery_rate(self) -> float:
        """Calculate recovery success rate"""
        if self.total_failures == 0:
            return 1.0
        return self.successful_recoveries / self.total_failures
        
    def get_stats(self) -> Dict:
        """Get comprehensive recovery statistics"""
        mttr = self.get_mttr()
        
        return {
            "mttr": f"{mttr:.2f}s" if mttr else "N/A",
            "recovery_rate": f"{self.get_recovery_rate():.2%}",
            "total_failures": self.total_failures,
            "successful_recoveries": self.successful_recoveries,
            "failure_types": self._get_failure_distribution(),
            "recovery_methods": self._get_recovery_distribution()
        }
        
    def _get_failure_distribution(self) -> Dict[str, int]:
        """Get distribution of failure types"""
        distribution = {}
        for event in self.events:
            distribution[event.failure_type] = distribution.get(event.failure_type, 0) + 1
        return distribution
        
    def _get_recovery_distribution(self) -> Dict[str, int]:
        """Get distribution of recovery methods"""
        distribution = {}
        for event in self.events:
            distribution[event.recovery_method] = distribution.get(event.recovery_method, 0) + 1
        return distribution 