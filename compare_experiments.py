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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    if 'accuracy' in df.columns:
        axes[0].bar(df.index, df['accuracy'], color='skyblue')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylim([df['accuracy'].min() * 0.9, min(1.0, df['accuracy'].max() * 1.1)])
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    if 'training_time' in df.columns:
        axes[1].bar(df.index, df['training_time'], color='salmon')
        axes[1].set_title('Training Time Comparison')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    plt.close()
    
    print(f"\nComparison chart saved as experiment_comparison.png")
    
    # Identify best model based on accuracy
    if 'accuracy' in df.columns:
        best_model = df['accuracy'].idxmax()
        print(f"\nBest model based on accuracy: {best_model} (Accuracy: {df.loc[best_model, 'accuracy']:.4f})")
    
    # Identify fastest model
    if 'training_time' in df.columns:
        fastest_model = df['training_time'].idxmin()
        print(f"Fastest model: {fastest_model} (Training time: {df.loc[fastest_model, 'training_time']:.2f} seconds)")
    
    # Show advanced metrics for experiments that have them
    print("\n===== Additional Metrics =====")
    for exp_name, metrics in metrics.items():
        print(f"\n{exp_name}:")
        
        if 'best_model' in metrics:
            print(f"Best model: {metrics['best_model']}")
            
        if 'best_params' in metrics:
            print(f"Best parameters: {metrics['best_params']}")
            
        if 'original_features' in metrics and 'engineered_features' in metrics:
            print(f"Features: {metrics['original_features']} original -> {metrics['engineered_features']} after engineering")
            
        if 'model_performances' in metrics:
            print("Model performances:")
            if isinstance(metrics['model_performances'], dict):
                for model_name, perf in metrics['model_performances'].items():
                    performance_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                              for k, v in perf.items()])
                    print(f"  - {model_name}: {performance_str}")

if __name__ == "__main__":
    # Experiment directories to compare
    experiment_dirs = [
        ".",  # Base experiment
        "../git-worktree-example-experiments/experiment_hyperparameter",
        "../git-worktree-example-experiments/experiment_feature_engineering",
        "../git-worktree-example-experiments/experiment_model_selection",
        "../git-worktree-example-experiments/experiment_ensemble"  # New ensemble experiment
    ]
    
    # Load and compare metrics
    metrics = load_metrics(experiment_dirs)
    
    if metrics:
        compare_metrics(metrics)
    else:
        print("No metrics found to compare. Please run the experiments first.") 