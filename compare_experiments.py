import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(experiment_dirs):
    """Load metrics from all experiment directories"""
    all_metrics = {}
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                all_metrics[exp_name] = metrics
        else:
            print(f"Warning: No metrics.json found in {exp_dir}")
    
    return all_metrics

def compare_metrics(metrics):
    """Compare metrics from different experiments"""
    # Create DataFrame for comparison
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Print comparison table
    print("\n===== Experiment Comparison =====")
    print(df.to_string())
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    ax1.bar(df.index, df['accuracy'], color='skyblue')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim([df['accuracy'].min() * 0.9, min(1.0, df['accuracy'].max() * 1.1)])
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Training time comparison
    ax2.bar(df.index, df['training_time'], color='salmon')
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    plt.close()
    
    print(f"\nComparison chart saved as experiment_comparison.png")
    
    # Identify best model based on accuracy
    best_model = df['accuracy'].idxmax()
    print(f"\nBest model based on accuracy: {best_model} (Accuracy: {df.loc[best_model, 'accuracy']:.4f})")
    
    # Identify fastest model
    fastest_model = df['training_time'].idxmin()
    print(f"Fastest model: {fastest_model} (Training time: {df.loc[fastest_model, 'training_time']:.2f} seconds)")

if __name__ == "__main__":
    # Experiment directories to compare
    experiment_dirs = [
        ".",  # Base experiment
        "../git-worktree-example-experiments/experiment_hyperparameter",
        "../git-worktree-example-experiments/experiment_feature_engineering",
        "../git-worktree-example-experiments/experiment_model_selection"
    ]
    
    # Load and compare metrics
    metrics = load_metrics(experiment_dirs)
    
    if metrics:
        compare_metrics(metrics)
    else:
        print("No metrics found to compare. Please run the experiments first.") 