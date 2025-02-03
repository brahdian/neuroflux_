import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import torch
from datetime import datetime, timedelta
import wandb
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import threading
from functools import lru_cache
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

class NeuroFluxVisualizer:
    """Memory-efficient visualization with streaming"""
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe components
        self.update_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.cache = LRUCache(maxsize=100)
        self.metric_buffers = {}
        self.running = True
        
        # Chunked data handling
        self.chunk_size = 1000
        
    def create_training_dashboard(
        self,
        metrics: Dict[str, List[float]],
        step_size: int = 100
    ) -> go.Figure:
        """Create dashboard with streaming data"""
        # Create base figure
        fig = self._create_base_dashboard()
        
        # Stream data in chunks
        for metric, values in metrics.items():
            for chunk in self._iter_chunks(values):
                self._update_plot(
                    fig,
                    metric,
                    chunk,
                    step_size
                )
                
        return fig
        
    def _iter_chunks(self, data: List[float]):
        """Iterate over data in chunks"""
        for i in range(0, len(data), self.chunk_size):
            yield data[i:i + self.chunk_size]
            
    def create_interactive_report(
        self,
        training_data: Dict,
        monitor_data: Dict,
        save_path: Optional[str] = None
    ) -> HTML:
        """Create paginated interactive report"""
        # Generate visualizations with chunking
        dashboard = self._create_chunked_dashboard(training_data)
        heatmap = self._create_chunked_heatmap(monitor_data)
        analysis = self._create_chunked_analysis(training_data)
        
        # Create paginated HTML
        html_content = self._create_paginated_html(
            dashboard,
            heatmap,
            analysis
        )
        
        if save_path:
            self._save_chunked_html(html_content, save_path)
            
        return HTML(html_content)
        
    def _create_paginated_html(self, *sections):
        """Create HTML with pagination"""
        return f"""
        <html>
            <head>
                <title>NeuroFlux Training Report</title>
                <script>
                    // Lazy loading logic
                    {self._get_lazy_loading_js()}
                </script>
            </head>
            <body>
                <div class="pagination">
                    {self._create_pagination_controls()}
                </div>
                <div class="content">
                    {self._create_paginated_content(sections)}
                </div>
            </body>
        </html>
        """
    
    def create_resource_heatmap(
        self,
        monitor_data: Dict[str, List[float]],
        interval_minutes: int = 5
    ) -> go.Figure:
        """Create resource utilization heatmap"""
        # Prepare data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq=f'{interval_minutes}min'
        )
        
        metrics = ['gpu_util', 'gpu_memory', 'cpu_util', 'memory_util']
        data = np.zeros((len(metrics), len(timestamps)))
        
        for i, metric in enumerate(metrics):
            if metric in monitor_data:
                # Resample data to match timestamps
                values = monitor_data[metric]
                data[i, :] = np.interp(
                    np.linspace(0, 1, len(timestamps)),
                    np.linspace(0, 1, len(values)),
                    values
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=timestamps,
            y=metrics,
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title='Resource Utilization Heatmap',
            xaxis_title='Time',
            yaxis_title='Metric',
            height=400
        )
        
        # Save heatmap
        fig.write_html(self.save_dir / "resource_heatmap.html")
        return fig
    
    def create_performance_analysis(
        self,
        training_metrics: Dict[str, List[float]],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """Create performance analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Distribution',
                'Correlation Matrix',
                'Throughput vs. Batch Size',
                'Memory vs. Sequence Length'
            )
        )
        
        # Performance distribution
        for i, (metric, values) in enumerate(training_metrics.items()):
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=metric,
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Correlation matrix
        df = pd.DataFrame(training_metrics)
        corr = df.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu'
            ),
            row=1, col=2
        )
        
        # Throughput vs. Batch Size
        if all(k in training_metrics for k in ['throughput', 'batch_size']):
            fig.add_trace(
                go.Scatter(
                    x=training_metrics['batch_size'],
                    y=training_metrics['throughput'],
                    mode='markers',
                    name='Throughput'
                ),
                row=2, col=1
            )
        
        # Memory vs. Sequence Length
        if all(k in training_metrics for k in ['gpu_memory', 'seq_length']):
            fig.add_trace(
                go.Scatter(
                    x=training_metrics['seq_length'],
                    y=training_metrics['gpu_memory'],
                    mode='markers',
                    name='Memory'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text="Performance Analysis"
        )
        
        # Save analysis
        fig.write_html(self.save_dir / "performance_analysis.html")
        return fig
    
    def _smooth_curve(
        self,
        values: List[float],
        weight: float = 0.6
    ) -> List[float]:
        """Apply exponential smoothing to curve"""
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def export_wandb_artifacts(
        self,
        run_id: Optional[str] = None
    ):
        """Export visualizations to W&B"""
        if wandb.run is None and run_id is None:
            raise ValueError("No active W&B run found")
            
        # Log all HTML artifacts
        for html_file in self.save_dir.glob("*.html"):
            artifact = wandb.Artifact(
                name=f"visualization_{html_file.stem}",
                type="visualization"
            )
            artifact.add_file(str(html_file))
            wandb.log_artifact(artifact)

    def update_metrics(self, metrics: Dict[str, float]):
        """Thread-safe metric updates"""
        with self.update_lock:
            current_time = time.time()
            for metric, value in metrics.items():
                if metric not in self.metric_buffers:
                    self.metric_buffers[metric] = deque(maxlen=1000)
                self.metric_buffers[metric].append((current_time, value))
    
    def _update_plots(self):
        """Thread-safe plot updates"""
        if not self.running:
            return
            
        try:
            with self.update_lock:
                with self.fig.batch_update():
                    for i, (metric, buffer) in enumerate(self.metric_buffers.items()):
                        if buffer:
                            times, values = zip(*list(buffer))
                            self.fig.data[i].x = times
                            self.fig.data[i].y = values
        except Exception as e:
            logger.error(f"Plot update failed: {str(e)}")

class LiveVisualizer:
    """Real-time visualization for NeuroFlux training"""
    def __init__(self, update_interval: float = 1.0):
        # Setup live plotting
        self.fig = go.FigureWidget()
        self.update_interval = update_interval
        
        # Metric buffers with fixed size
        self.buffer_size = 1000
        self.metric_buffers = {
            'loss': deque(maxlen=self.buffer_size),
            'throughput': deque(maxlen=self.buffer_size),
            'gpu_util': deque(maxlen=self.buffer_size),
            'memory_util': deque(maxlen=self.buffer_size),
            'raid_health': deque(maxlen=self.buffer_size),
            'expert_balance': deque(maxlen=self.buffer_size)
        }
        
        # Initialize plots
        self._setup_live_plots()
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
    def _setup_live_plots(self):
        """Setup interactive subplot layout"""
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Training Loss', 'Throughput',
                'GPU & Memory', 'Expert Balance',
                'RAID Health', 'System Overview'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Initialize empty traces
        for i, metric in enumerate(self.metric_buffers.keys()):
            if metric != 'system_health':
                self.fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        name=metric,
                        mode='lines'
                    ),
                    row=(i//2)+1,
                    col=(i%2)+1
                )
        
        # Add system health indicator
        self.fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=100,
                domain={'row': 2, 'column': 1},
                title={'text': "System Health"}
            ),
            row=3, col=2
        )
        
        # Update layout
        self.fig.update_layout(
            height=800,
            showlegend=True,
            title_text="NeuroFlux Live Training Monitor",
            uirevision=True  # Preserve zoom on updates
        )
        
    def _update_loop(self):
        """Continuous plot update loop"""
        while self.running:
            try:
                self._update_plots()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Live visualization error: {e}")
                
    def _update_plots(self):
        """Update all plot traces"""
        with self.fig.batch_update():
            for i, (metric, buffer) in enumerate(self.metric_buffers.items()):
                if buffer and metric != 'system_health':
                    times, values = zip(*list(buffer))
                    self.fig.data[i].x = times
                    self.fig.data[i].y = values
                    
            # Update system health indicator
            system_health = self._compute_system_health()
            self.fig.data[-1].value = system_health
            
    def _compute_system_health(self) -> float:
        """Compute overall system health score"""
        health_factors = {
            'gpu_util': (0.7, 0.95),  # (optimal, max)
            'memory_util': (0.6, 0.9),
            'raid_health': (0.8, 1.0),
            'expert_balance': (0.7, 1.0)
        }
        
        scores = []
        for metric, (opt, max_val) in health_factors.items():
            if self.metric_buffers[metric]:
                _, latest = self.metric_buffers[metric][-1]
                score = 1.0 - (abs(latest - opt) / (max_val - opt))
                scores.append(max(0.0, min(1.0, score)))
                
        return 100 * (sum(scores) / len(scores)) if scores else 100
        
    def display(self):
        """Display live visualization"""
        display(self.fig)
        
    def cleanup(self):
        """Cleanup visualization resources"""
        self.running = False
        self.update_thread.join(timeout=5.0) 