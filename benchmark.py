import time
import os
import sys
import json
import psutil
import traceback
import tracemalloc
from contextlib import contextmanager
from train import train_model

class BenchmarkResults:
    def __init__(self):
        self.execution_times = []
        self.memory_peaks = []
        self.cpu_percentages = []
        
    def add_run(self, execution_time, memory_peak, cpu_percentage):
        self.execution_times.append(execution_time)
        self.memory_peaks.append(memory_peak)
        self.cpu_percentages.append(cpu_percentage)
    
    def to_dict(self):
        if not self.execution_times:
            return {}
            
        return {
            "execution_time": {
                "mean": sum(self.execution_times) / len(self.execution_times),
                "min": min(self.execution_times),
                "max": max(self.execution_times),
                "values": self.execution_times
            },
            "memory_peak": {
                "mean": sum(self.memory_peaks) / len(self.memory_peaks),
                "min": min(self.memory_peaks),
                "max": max(self.memory_peaks),
                "values": self.memory_peaks
            },
            "cpu_percentage": {
                "mean": sum(self.cpu_percentages) / len(self.cpu_percentages),
                "min": min(self.cpu_percentages),
                "max": max(self.cpu_percentages),
                "values": self.cpu_percentages
            }
        }

def run_with_performance_tracking():
    """Run the training with performance tracking"""
    # Start memory tracking
    tracemalloc.start()
    
    # Get the process to monitor CPU
    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()
    
    # Time tracking
    start_time = time.time()
    
    # Run the model training
    metrics = train_model()
    
    # Time measurement
    execution_time = time.time() - start_time
    
    # Memory measurement
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # CPU measurement
    end_cpu_times = process.cpu_times()
    cpu_user_time = end_cpu_times.user - start_cpu_times.user
    cpu_system_time = end_cpu_times.system - start_cpu_times.system
    cpu_total_time = cpu_user_time + cpu_system_time
    
    # Avoid division by zero
    if execution_time > 0:
        cpu_percentage = cpu_total_time / execution_time * 100
    else:
        cpu_percentage = 0
    
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Memory peak: {peak / (1024 * 1024):.2f} MB")
    print(f"CPU usage: {cpu_percentage:.2f}%")
    
    return metrics, execution_time, peak / (1024 * 1024), cpu_percentage

def run_benchmark(num_runs=3):
    benchmark_results = BenchmarkResults()
    
    for i in range(num_runs):
        print(f"\nBenchmark run {i+1}/{num_runs}")
        try:
            metrics, execution_time, memory_peak, cpu_percentage = run_with_performance_tracking()
            benchmark_results.add_run(execution_time, memory_peak, cpu_percentage)
        except Exception as e:
            print(f"Run failed: {e}")
            traceback.print_exc()
    
    return benchmark_results

if __name__ == "__main__":
    print("Running performance benchmark...")
    benchmark_results = run_benchmark(num_runs=2)
    
    # Combine benchmark results with model metrics
    try:
        with open('metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except FileNotFoundError:
        model_metrics = {}
    
    # Add benchmark data to metrics
    model_metrics['benchmark'] = benchmark_results.to_dict()
    
    # Save updated metrics
    with open('benchmark_metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    print("\nBenchmark complete.")
    print(f"Results saved to benchmark_metrics.json") 