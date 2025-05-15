#!/usr/bin/env python3
"""
Identifies the best algorithm branch based on benchmark metrics and optionally merges it.
Part of the AlphaEvolve workflow's evaluator pool and selection policy.
"""

import argparse
import json
import os
import glob
import subprocess
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a shell command and return the output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        if capture_output:
            print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip() if capture_output else True


def load_metrics_file(file_path):
    """Load metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_metrics_files(benchmark_dir, pattern='benchmark_metrics.json'):
    """Find all metrics files in the benchmark directory."""
    if not os.path.isdir(benchmark_dir):
        print(f"Benchmark directory '{benchmark_dir}' does not exist.")
        return []
    
    # Look for metrics files in subdirectories
    return glob.glob(os.path.join(benchmark_dir, f"**/{pattern}"), recursive=True)


def find_best_branch(benchmark_dir, metric='accuracy', reverse=True):
    """
    Find the best branch based on the specified metric.
    
    Args:
        benchmark_dir: Directory containing benchmark results
        metric: Metric to use for comparison (default: accuracy)
        reverse: If True, higher is better (e.g., accuracy); if False, lower is better (e.g., time)
        
    Returns:
        Tuple of (best_branch, best_value)
    """
    metrics_files = find_metrics_files(benchmark_dir)
    
    if not metrics_files:
        print(f"No metrics files found in '{benchmark_dir}'")
        return None, None
    
    best_branch = None
    best_value = float('-inf') if reverse else float('inf')
    
    for file_path in metrics_files:
        # Extract branch name from directory structure
        # Expected format: benchmark_dir/metrics-{branch}/benchmark_metrics.json
        branch = os.path.basename(os.path.dirname(file_path))
        if branch.startswith('metrics-'):
            branch = branch[len('metrics-'):]
        
        metrics = load_metrics_file(file_path)
        if not metrics:
            continue
        
        # Check if the metric exists in this file
        if metric in metrics:
            value = float(metrics[metric])
            print(f"Branch {branch}: {metric} = {value}")
            
            if (reverse and value > best_value) or (not reverse and value < best_value):
                best_value = value
                best_branch = branch
    
    return best_branch, best_value


def merge_branch(branch, target_branch, message=None):
    """Merge the specified branch into the target branch."""
    # Make sure we're on the target branch
    if not run_command(['git', 'checkout', target_branch]):
        return False
    
    # Create a merge commit
    merge_cmd = ['git', 'merge', '--no-ff', branch]
    if message:
        merge_cmd.extend(['-m', message])
    
    return run_command(merge_cmd, capture_output=False)


def main():
    parser = argparse.ArgumentParser(description='Find and merge the best algorithm branch')
    parser.add_argument('--benchmark-dir', required=True, help='Directory containing benchmark results')
    parser.add_argument('--target-branch', default='main', help='Branch to merge into (default: main)')
    parser.add_argument('--metric', default='accuracy', help='Metric to use for selecting the best branch (default: accuracy)')
    parser.add_argument('--reverse', action='store_true', default=True, help='Whether higher values are better (default: True)')
    parser.add_argument('--dry-run', action='store_true', help='Only identify the best branch without merging')
    
    args = parser.parse_args()
    
    # Adjust reverse flag based on metric
    if args.metric in ['training_time', 'inference_time', 'memory_usage', 'error_rate']:
        args.reverse = False
    
    # Find the best branch
    best_branch, best_value = find_best_branch(args.benchmark_dir, args.metric, args.reverse)
    
    if not best_branch:
        print("No best branch found.")
        return 1
    
    print(f"\nBest branch based on {args.metric}: {best_branch} (value: {best_value})")
    
    # Merge if not dry run
    if not args.dry_run:
        print(f"\nMerging {best_branch} into {args.target_branch}...")
        
        message = f"Merge branch '{best_branch}' as best algorithm candidate\n\n" \
                 f"- {args.metric}: {best_value}\n" \
                 f"- Selected by AlphaEvolve evaluator pool"
        
        if merge_branch(best_branch, args.target_branch, message):
            print(f"Successfully merged {best_branch} into {args.target_branch}")
            return 0
        else:
            print(f"Failed to merge {best_branch}")
            return 1
    else:
        print(f"Dry run - would merge {best_branch} into {args.target_branch}")
        return 0


if __name__ == "__main__":
    exit(main()) 