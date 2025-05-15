#!/usr/bin/env python3
"""
Merge Best Branch - AlphaEvolve Selection & Merge Policy

This script:
1. Loads benchmark results from all candidate branches
2. Identifies the best-performing branch based on a configurable metric
3. Automatically merges the best branch back to the target branch (e.g., main/experiment_ensemble)

Usage:
  python merge_best_branch.py --target-branch experiment_ensemble --metric accuracy
"""

import os
import glob
import json
import argparse
import subprocess
import pandas as pd

def load_metrics_file(filepath):
    """Load metrics JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def run_command(command, capture_output=True):
    """Run a shell command and return output."""
    result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
    if result.returncode != 0 and capture_output:
        print(f"Command failed: {command}")
        print(f"Error: {result.stderr}")
    return result

def get_branch_metrics(benchmark_dirs):
    """Get metrics for all branches from benchmark directories."""
    branch_metrics = {}
    
    for benchmark_dir in benchmark_dirs:
        # Extract branch name from directory name
        branch_name = os.path.basename(benchmark_dir).split('-', 1)[1]
        metrics_file = os.path.join(benchmark_dir, 'benchmark_metrics.json')
        
        metrics = load_metrics_file(metrics_file)
        if metrics:
            # Extract key metrics
            branch_metrics[branch_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'training_time': metrics.get('training_time', float('inf')),
                'best_model': metrics.get('best_model', 'unknown')
            }
            
            # Get additional metrics from benchmark section if available
            if 'benchmark' in metrics:
                benchmark = metrics['benchmark']
                if 'execution_time' in benchmark:
                    branch_metrics[branch_name]['execution_time'] = benchmark['execution_time'].get('mean', float('inf'))
                if 'memory_peak' in benchmark:
                    branch_metrics[branch_name]['memory_peak'] = benchmark['memory_peak'].get('mean', float('inf'))
    
    return branch_metrics

def find_best_branch(metrics, metric_name='accuracy', higher_is_better=True):
    """Find the best branch based on the specified metric."""
    if not metrics:
        return None
    
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Check if metric exists
    if metric_name not in df.columns:
        print(f"Metric '{metric_name}' not found. Available metrics: {list(df.columns)}")
        return None
    
    # Find best branch
    if higher_is_better:
        best_branch = df[metric_name].idxmax()
    else:
        best_branch = df[metric_name].idxmin()
    
    return best_branch, df.loc[best_branch].to_dict()

def merge_branch(branch_name, target_branch):
    """Merge the specified branch into the target branch."""
    # Checkout target branch
    result = run_command(f"git checkout {target_branch}")
    if result.returncode != 0:
        print(f"Failed to checkout {target_branch}")
        return False
    
    # Merge branch with no fast-forward
    message = f"Merge branch '{branch_name}' - AlphaEvolve automated selection"
    result = run_command(f"git merge --no-ff {branch_name} -m \"{message}\"")
    
    if result.returncode != 0:
        print(f"Failed to merge {branch_name} into {target_branch}. Manual intervention needed.")
        return False
    
    print(f"Successfully merged {branch_name} into {target_branch}")
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AlphaEvolve - Merge best branch based on metrics")
    parser.add_argument("--benchmark-dir", default="benchmark-results", help="Directory containing benchmark results")
    parser.add_argument("--target-branch", default="experiment_ensemble", help="Branch to merge into")
    parser.add_argument("--metric", default="accuracy", help="Metric to use for selection")
    parser.add_argument("--higher-is-better", action="store_true", default=True, help="Whether higher metric values are better")
    parser.add_argument("--dry-run", action="store_true", help="Only identify best branch without merging")
    
    args = parser.parse_args()
    
    # Find benchmark directories
    benchmark_dirs = glob.glob(f"{args.benchmark_dir}/benchmark-*")
    if not benchmark_dirs:
        print(f"No benchmark results found in {args.benchmark_dir}")
        return 1
    
    # Get metrics for all branches
    branch_metrics = get_branch_metrics(benchmark_dirs)
    
    # Find best branch
    result = find_best_branch(branch_metrics, args.metric, args.higher_is_better)
    if result is None:
        print("No best branch found")
        return 1
    
    best_branch, best_metrics = result
    
    # Print result
    print(f"Best branch: {best_branch}")
    for metric, value in best_metrics.items():
        print(f"- {metric}: {value}")
    
    # Merge if not dry run
    if args.dry_run:
        print("Dry run - not merging")
        return 0
    
    # Merge best branch
    success = merge_branch(best_branch, args.target_branch)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 