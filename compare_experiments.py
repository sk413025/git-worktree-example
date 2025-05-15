#!/usr/bin/env python3
"""
Compare experiment results across different algorithm implementations.
Used to evaluate AlphaEvolve candidate branches and visualize performance.
"""

import argparse
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics_from_file(file_path):
    """Load metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def load_metrics_from_directory(directory, filename='benchmark_metrics.json'):
    """Load metrics from all subdirectories in the given directory."""
    metrics_files = glob.glob(os.path.join(directory, f"*//{filename}"))
    
    results = {}
    for file_path in metrics_files:
        # Extract experiment name from directory structure
        experiment_name = os.path.basename(os.path.dirname(file_path))
        results[experiment_name] = load_metrics_from_file(file_path)
    
    return results


def load_metrics_from_worktrees(base_dir="../", prefix="candidate-", filename='benchmark_metrics.json'):
    """Load metrics from git worktree directories."""
    # Find directories matching the pattern
    candidate_dirs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    
    results = {}
    for dir_name in candidate_dirs:
        # Extract branch name from directory name
        branch_name = dir_name.replace(prefix, "")
        
        # Load metrics file
        metrics_file = os.path.join(base_dir, dir_name, filename)
        if os.path.exists(metrics_file):
            results[branch_name] = load_metrics_from_file(metrics_file)
        else:
            print(f"No metrics file found for {branch_name}")
    
    return results


def create_comparison_dataframe(metrics_dict, key_metrics=None):
    """Convert metrics dictionary to pandas DataFrame for comparison."""
    if not metrics_dict:
        return pd.DataFrame()
    
    # Determine metrics to extract if not specified
    if key_metrics is None:
        # Get all keys from the first experiment as a starting point
        first_exp = next(iter(metrics_dict.values()))
        key_metrics = list(first_exp.keys())
    
    # Create a list to hold the data
    data = []
    
    for exp_name, metrics in metrics_dict.items():
        row = {'experiment': exp_name}
        
        # Extract metrics that exist
        for metric in key_metrics:
            if metric in metrics:
                row[metric] = metrics[metric]
        
        data.append(row)
    
    # Convert to DataFrame
    return pd.DataFrame(data)


def plot_metrics_comparison(df, metrics_to_plot, output_file=None, title="Algorithm Performance Comparison"):
    """Create bar charts comparing key metrics across experiments."""
    if df.empty or not metrics_to_plot:
        print("No data to plot")
        return
    
    # Get actual metrics columns that exist in the dataframe
    existing_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not existing_metrics:
        print("None of the specified metrics exist in the data")
        return
    
    # Set up the figure with subplots
    n_metrics = len(existing_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    # Handle case with single metric
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(existing_metrics):
        # Skip if metric doesn't exist
        if metric not in df.columns:
            continue
            
        # Sort by metric value (descending for accuracy/F1, ascending for time/errors)
        ascending = metric in ['training_time', 'inference_time', 'memory_usage', 'error_rate']
        sorted_df = df.sort_values(by=metric, ascending=ascending)
        
        # Plot bar chart
        ax = axes[i]
        bars = sorted_df.plot(kind='bar', x='experiment', y=metric, ax=ax, 
                              legend=False, colormap='viridis')
        
        # Add value labels on top of bars
        for j, p in enumerate(ax.patches):
            value = p.get_height()
            if not pd.isna(value):  # Skip if NaN
                # Format based on metric type
                if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                    value_text = f"{value:.3f}"
                elif metric in ['training_time', 'inference_time']:
                    value_text = f"{value:.2f}s"
                elif metric == 'memory_usage':
                    value_text = f"{value:.1f}MB"
                else:
                    value_text = f"{value}"
                
                ax.annotate(value_text, 
                           (p.get_x() + p.get_width() / 2., value),
                           ha='center', va='bottom', rotation=0, 
                           xytext=(0, 5), textcoords='offset points')
        
        # Customize appearance
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title and adjust layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for suptitle
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def calculate_overall_score(df, weights=None):
    """Calculate an overall score for each experiment based on weighted metrics."""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default weights if none provided
    if weights is None:
        weights = {
            'accuracy': 1.0,
            'f1_score': 0.8,
            'precision': 0.5,
            'recall': 0.5,
            'training_time': -0.3,  # Negative weight for time (lower is better)
            'inference_time': -0.5,
            'memory_usage': -0.3,
        }
    
    # Initialize total score column
    result_df['total_score'] = 0.0
    
    # Calculate normalized contributions for each metric
    for metric, weight in weights.items():
        if metric in result_df.columns:
            # Skip if all values are NaN
            if result_df[metric].isna().all():
                continue
                
            # For metrics where lower is better, invert the values
            if weight < 0:
                # Replace 0 with small value to avoid division by zero
                min_val = result_df[metric].replace(0, 1e-10).min()
                result_df[f'{metric}_norm'] = min_val / result_df[metric].replace(0, 1e-10)
                result_df['total_score'] += abs(weight) * result_df[f'{metric}_norm']
            else:
                # For metrics where higher is better, normalize to 0-1 range
                max_val = result_df[metric].max()
                if max_val > 0:
                    result_df[f'{metric}_norm'] = result_df[metric] / max_val
                    result_df['total_score'] += weight * result_df[f'{metric}_norm']
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results across algorithm implementations')
    parser.add_argument('--dir', type=str, default='.', 
                      help='Directory containing experiment results')
    parser.add_argument('--worktrees', action='store_true',
                      help='Look for metrics in git worktree directories')
    parser.add_argument('--base-dir', type=str, default='../',
                      help='Base directory for worktrees (default: ../)')
    parser.add_argument('--prefix', type=str, default='candidate-',
                      help='Prefix for worktree directories (default: candidate-)')
    parser.add_argument('--metrics', type=str, nargs='+',
                      default=['accuracy', 'f1_score', 'training_time', 'inference_time', 'memory_usage'],
                      help='Metrics to compare')
    parser.add_argument('--output', type=str, default='experiment_comparison.png',
                      help='Output file for comparison chart')
    parser.add_argument('--weights', type=str, default=None,
                      help='JSON file with metric weights for overall score')
    
    args = parser.parse_args()
    
    # Load metrics data
    metrics_dict = {}
    if args.worktrees:
        print(f"Loading metrics from worktree directories in {args.base_dir}")
        metrics_dict = load_metrics_from_worktrees(args.base_dir, args.prefix)
    else:
        print(f"Loading metrics from subdirectories in {args.dir}")
        metrics_dict = load_metrics_from_directory(args.dir)
    
    if not metrics_dict:
        print("No metrics data found")
        return
    
    # Create DataFrame for comparison
    df = create_comparison_dataframe(metrics_dict)
    
    # Load custom weights if provided
    weights = None
    if args.weights:
        try:
            with open(args.weights, 'r') as f:
                weights = json.load(f)
        except Exception as e:
            print(f"Error loading weights file: {e}")
    
    # Calculate overall score
    df = calculate_overall_score(df, weights)
    
    # Print summary table
    pd.set_option('display.max_columns', None)
    print("\nExperiment Results Summary:")
    print(df[['experiment', 'accuracy', 'f1_score', 'training_time', 'memory_usage', 'total_score']].to_string(index=False))
    
    # Determine the best experiment
    best_exp = df.loc[df['total_score'].idxmax()]
    print(f"\nBest experiment: {best_exp['experiment']} (Score: {best_exp['total_score']:.4f})")
    
    # Plot comparison
    plot_metrics_comparison(df, args.metrics, args.output)
    
    # Save the results to a JSON file
    output_json = Path(args.output).with_suffix('.json')
    df.to_json(output_json, orient='records', indent=2)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    main() 