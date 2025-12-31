#!/usr/bin/env python3
"""
Resource Usage Monitor

Monitors system resource usage during video analysis to prevent memory leaks
and ensure stable performance under various conditions.
"""

import time
import psutil
import threading
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import cv2
import gc


class ResourceMonitor:
    """Monitors system resources and provides optimization recommendations."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Resource tracking
        self.cpu_usage = deque(maxlen=60)  # Last 60 seconds
        self.memory_usage = deque(maxlen=60)
        self.memory_available = deque(maxlen=60)
        self.frame_rates = deque(maxlen=30)  # Last 30 measurements
        
        # Thresholds for warnings
        self.cpu_warning_threshold = 80.0  # %
        self.memory_warning_threshold = 85.0  # %
        self.fps_warning_threshold = 15.0  # FPS
        
        # Process reference
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                system_memory = psutil.virtual_memory()
                
                self.memory_usage.append(memory_percent)
                self.memory_available.append(system_memory.available / 1024 / 1024 / 1024)  # GB
                
                # Check for warnings
                self._check_resource_warnings(cpu_percent, memory_percent)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_resource_warnings(self, cpu_percent: float, memory_percent: float):
        """Check for resource usage warnings."""
        if cpu_percent > self.cpu_warning_threshold:
            print(f"⚠️  High CPU usage: {cpu_percent:.1f}%")
        
        if memory_percent > self.memory_warning_threshold:
            print(f"⚠️  High memory usage: {memory_percent:.1f}%")
            # Suggest garbage collection
            gc.collect()
    
    def record_frame_rate(self, fps: float):
        """Record frame rate measurement."""
        self.frame_rates.append(fps)
        
        if fps < self.fps_warning_threshold:
            print(f"⚠️  Low frame rate: {fps:.1f} FPS")
    
    def get_current_stats(self) -> Dict:
        """Get current resource usage statistics."""
        try:
            # Current values
            cpu_current = psutil.cpu_percent(interval=None)
            memory_info = self.process.memory_info()
            memory_current = self.process.memory_percent()
            
            # Averages
            cpu_avg = np.mean(self.cpu_usage) if self.cpu_usage else 0
            memory_avg = np.mean(self.memory_usage) if self.memory_usage else 0
            fps_avg = np.mean(self.frame_rates) if self.frame_rates else 0
            
            return {
                'cpu': {
                    'current': cpu_current,
                    'average': cpu_avg,
                    'peak': max(self.cpu_usage) if self.cpu_usage else 0
                },
                'memory': {
                    'current_mb': memory_info.rss / 1024 / 1024,
                    'current_percent': memory_current,
                    'average_percent': memory_avg,
                    'peak_mb': max([self.process.memory_info().rss / 1024 / 1024 
                                  for _ in range(1)]) if self.memory_usage else 0
                },
                'fps': {
                    'current': self.frame_rates[-1] if self.frame_rates else 0,
                    'average': fps_avg,
                    'min': min(self.frame_rates) if self.frame_rates else 0
                },
                'system': {
                    'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
                }
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def generate_performance_report(self) -> str:
        """Generate a performance report with recommendations."""
        stats = self.get_current_stats()
        
        if not stats:
            return "Unable to generate performance report"
        
        report = []
        report.append("=== Performance Report ===")
        
        # CPU Analysis
        report.append(f"\n🖥️  CPU Usage:")
        report.append(f"   Current: {stats['cpu']['current']:.1f}%")
        report.append(f"   Average: {stats['cpu']['average']:.1f}%")
        report.append(f"   Peak: {stats['cpu']['peak']:.1f}%")
        
        if stats['cpu']['average'] > 70:
            report.append("   ⚠️  High CPU usage detected")
            report.append("   💡 Recommendation: Reduce processing complexity or frame rate")
        elif stats['cpu']['average'] < 30:
            report.append("   ✅ CPU usage is optimal")
        
        # Memory Analysis
        report.append(f"\n💾 Memory Usage:")
        report.append(f"   Current: {stats['memory']['current_mb']:.1f}MB ({stats['memory']['current_percent']:.1f}%)")
        report.append(f"   Average: {stats['memory']['average_percent']:.1f}%")
        report.append(f"   Available: {stats['system']['available_memory_gb']:.1f}GB")
        
        if stats['memory']['current_percent'] > 80:
            report.append("   ⚠️  High memory usage detected")
            report.append("   💡 Recommendation: Enable garbage collection optimization")
        elif stats['memory']['current_mb'] > 1000:  # > 1GB
            report.append("   ⚠️  Large memory footprint")
            report.append("   💡 Recommendation: Check for memory leaks")
        else:
            report.append("   ✅ Memory usage is acceptable")
        
        # FPS Analysis
        report.append(f"\n📊 Frame Rate:")
        report.append(f"   Current: {stats['fps']['current']:.1f} FPS")
        report.append(f"   Average: {stats['fps']['average']:.1f} FPS")
        report.append(f"   Minimum: {stats['fps']['min']:.1f} FPS")
        
        if stats['fps']['average'] < 15:
            report.append("   ⚠️  Low frame rate detected")
            report.append("   💡 Recommendations:")
            report.append("      - Reduce camera resolution")
            report.append("      - Disable non-essential processing")
            report.append("      - Check camera drivers")
        elif stats['fps']['average'] >= 25:
            report.append("   ✅ Frame rate is good")
        else:
            report.append("   ⚠️  Frame rate could be improved")
        
        # System Health
        report.append(f"\n🏥 System Health:")
        memory_pressure = (stats['system']['total_memory_gb'] - stats['system']['available_memory_gb']) / stats['system']['total_memory_gb'] * 100
        report.append(f"   Memory pressure: {memory_pressure:.1f}%")
        
        if memory_pressure > 85:
            report.append("   ⚠️  System under memory pressure")
            report.append("   💡 Recommendation: Close other applications")
        else:
            report.append("   ✅ System memory is healthy")
        
        return "\n".join(report)
    
    def optimize_system_settings(self) -> List[str]:
        """Provide system optimization recommendations."""
        recommendations = []
        stats = self.get_current_stats()
        
        if not stats:
            return ["Unable to analyze system for optimization"]
        
        # CPU optimizations
        if stats['cpu']['average'] > 70:
            recommendations.extend([
                "Reduce MediaPipe model complexity",
                "Lower camera frame rate",
                "Disable non-essential visualizations",
                "Use frame skipping for processing"
            ])
        
        # Memory optimizations
        if stats['memory']['current_percent'] > 60:
            recommendations.extend([
                "Enable periodic garbage collection",
                "Use frame buffer reuse",
                "Reduce temporal smoother window sizes",
                "Clear unused landmark data"
            ])
        
        # FPS optimizations
        if stats['fps']['average'] < 20:
            recommendations.extend([
                "Reduce camera resolution to 320x240",
                "Set camera buffer size to 1",
                "Use MJPEG codec for camera",
                "Skip frames during processing bottlenecks"
            ])
        
        if not recommendations:
            recommendations.append("System is performing well - no optimizations needed")
        
        return recommendations


class MemoryLeakDetector:
    """Detects potential memory leaks in the application."""
    
    def __init__(self, check_interval: int = 50):  # Check every N frames
        self.check_interval = check_interval
        self.frame_count = 0
        self.memory_samples = []
        self.baseline_memory = None
        
    def check_frame(self):
        """Check memory usage for this frame."""
        self.frame_count += 1
        
        if self.frame_count % self.check_interval == 0:
            memory_info = psutil.Process().memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            
            self.memory_samples.append(current_memory)
            
            if self.baseline_memory is None:
                self.baseline_memory = current_memory
            
            # Check for memory leak
            if len(self.memory_samples) >= 5:
                recent_avg = np.mean(self.memory_samples[-3:])
                growth = recent_avg - self.baseline_memory
                
                if growth > 100:  # > 100MB growth
                    print(f"⚠️  Potential memory leak detected: {growth:.1f}MB growth")
                    print("   💡 Forcing garbage collection...")
                    gc.collect()
                    
                    # Update baseline after cleanup
                    post_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if post_gc_memory < recent_avg - 20:  # Significant cleanup
                        self.baseline_memory = post_gc_memory
                        print(f"   ✅ Memory cleaned up: {recent_avg - post_gc_memory:.1f}MB freed")
    
    def get_memory_trend(self) -> str:
        """Get memory usage trend analysis."""
        if len(self.memory_samples) < 3:
            return "Insufficient data for trend analysis"
        
        recent_samples = self.memory_samples[-5:]
        if len(recent_samples) < 2:
            return "Insufficient recent data"
        
        # Calculate trend
        x = np.arange(len(recent_samples))
        coeffs = np.polyfit(x, recent_samples, 1)
        slope = coeffs[0]  # MB per measurement
        
        if slope > 5:  # Growing > 5MB per measurement
            return f"⚠️  Memory usage increasing: +{slope:.1f}MB per {self.check_interval} frames"
        elif slope < -2:  # Decreasing
            return f"✅ Memory usage decreasing: {slope:.1f}MB per {self.check_interval} frames"
        else:
            return f"✅ Memory usage stable: {slope:.1f}MB per {self.check_interval} frames"


def test_resource_monitoring():
    """Test the resource monitoring system."""
    print("Testing Resource Monitoring System")
    
    monitor = ResourceMonitor(monitoring_interval=0.5)
    leak_detector = MemoryLeakDetector(check_interval=10)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate video processing workload
        print("Simulating video processing workload...")
        
        for frame_num in range(100):
            # Simulate frame processing
            time.sleep(0.03)  # ~30 FPS
            
            # Simulate some memory allocation
            dummy_data = np.random.rand(100, 100, 3)
            
            # Record frame rate
            fps = 1.0 / 0.03
            monitor.record_frame_rate(fps)
            
            # Check for memory leaks
            leak_detector.check_frame()
            
            # Print status every 20 frames
            if frame_num % 20 == 0:
                stats = monitor.get_current_stats()
                print(f"Frame {frame_num}: {stats['fps']['current']:.1f} FPS, "
                      f"{stats['memory']['current_mb']:.1f}MB, "
                      f"{stats['cpu']['current']:.1f}% CPU")
        
        # Generate final report
        print("\n" + monitor.generate_performance_report())
        print(f"\nMemory trend: {leak_detector.get_memory_trend()}")
        
        # Get optimization recommendations
        recommendations = monitor.optimize_system_settings()
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    test_resource_monitoring()