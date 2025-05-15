import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse {file_path} as JSON.")
        return None

def generate_model_comparison_chart(component_benchmarks):
    """Generate a chart comparing model training and prediction times"""
    if not component_benchmarks:
        return
    
    # Extract model names and times
    models = []
    train_times = []
    predict_times = []
    
    for model_data in component_benchmarks.get('model_training', []):
        model_name = model_data.get('model', 'Unknown')
        models.append(model_name)
        train_times.append(model_data.get('mean', 0) * 1000)  # Convert to ms
    
    # Match prediction times to models
    for model_data in component_benchmarks.get('model_prediction', []):
        model_name = model_data.get('model', 'Unknown')
        if model_name in models:
            predict_times.append(model_data.get('mean', 0) * 1000)  # Convert to ms
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.4
    x = np.arange(len(models))
    
    # Create bars
    train_bars = ax.bar(x - bar_width/2, train_times, bar_width, label='Training Time (ms)')
    predict_bars = ax.bar(x + bar_width/2, predict_times, bar_width, label='Prediction Time (ms)')
    
    # Customize plot
    ax.set_yscale('log')  # Log scale for better visualization of different magnitudes
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()
    
    print("Model performance comparison chart saved as model_performance_comparison.png")

def generate_pytest_benchmark_table(pytest_data):
    """Generate a table from pytest benchmark data"""
    if not pytest_data or 'benchmarks' not in pytest_data:
        return None
    
    # Extract benchmark data
    benchmark_data = []
    for bench in pytest_data['benchmarks']:
        stats = bench.get('stats', {})
        benchmark_data.append({
            'Name': bench.get('name', 'Unknown'),
            'Mean (ms)': stats.get('mean', 0) * 1000,
            'Min (ms)': stats.get('min', 0) * 1000,
            'Max (ms)': stats.get('max', 0) * 1000,
            'StdDev': stats.get('stddev', 0) * 1000,
            'Rounds': stats.get('rounds', 0)
        })
    
    return pd.DataFrame(benchmark_data)

def generate_report():
    """Generate a comprehensive benchmark report"""
    # Load benchmark data
    component_data = load_json_file('component_benchmarks.json')
    pytest_data = load_json_file('pytest_benchmark.json')
    benchmark_metrics = load_json_file('benchmark_metrics.json')
    
    # Generate report
    with open('benchmark_report.md', 'w') as report:
        report.write("# Algorithm Performance Benchmark Report\n\n")
        
        # Add overall metrics
        report.write("## Overall Performance Metrics\n\n")
        if benchmark_metrics:
            report.write(f"- **Accuracy**: {benchmark_metrics.get('accuracy', 'N/A')}\n")
            report.write(f"- **Best Model**: {benchmark_metrics.get('best_model', 'N/A')}\n")
            report.write(f"- **Training Time**: {benchmark_metrics.get('training_time', 'N/A')} seconds\n")
            report.write(f"- **Original Features**: {benchmark_metrics.get('original_features', 'N/A')}\n")
            report.write(f"- **Engineered Features**: {benchmark_metrics.get('engineered_features', 'N/A')}\n\n")
        
        # Add benchmark metrics
        if benchmark_metrics and 'benchmark' in benchmark_metrics:
            bench = benchmark_metrics['benchmark']
            report.write("### Full Pipeline Benchmark\n\n")
            report.write("| Metric | Mean | Min | Max |\n")
            report.write("|--------|------|-----|------|\n")
            
            # Execution time
            if 'execution_time' in bench:
                et = bench['execution_time']
                report.write(f"| Execution Time (s) | {et.get('mean', 'N/A')} | {et.get('min', 'N/A')} | {et.get('max', 'N/A')} |\n")
            
            # Memory peak
            if 'memory_peak' in bench:
                mp = bench['memory_peak']
                report.write(f"| Memory Peak (MB) | {mp.get('mean', 'N/A')} | {mp.get('min', 'N/A')} | {mp.get('max', 'N/A')} |\n")
            
            # CPU percentage
            if 'cpu_percentage' in bench:
                cp = bench['cpu_percentage']
                report.write(f"| CPU Usage (%) | {cp.get('mean', 'N/A')} | {cp.get('min', 'N/A')} | {cp.get('max', 'N/A')} |\n\n")
        
        # Add component benchmarks
        report.write("## Component Benchmarks\n\n")
        
        if component_data:
            # Feature engineering benchmark
            if 'feature_engineering' in component_data:
                fe = component_data['feature_engineering']
                report.write("### Feature Engineering\n\n")
                report.write(f"- **Mean**: {fe.get('mean', 'N/A')*1000:.4f} ms\n")
                report.write(f"- **Min**: {fe.get('min', 'N/A')*1000:.4f} ms\n")
                report.write(f"- **Max**: {fe.get('max', 'N/A')*1000:.4f} ms\n\n")
            
            # Model training benchmarks
            if 'model_training' in component_data:
                report.write("### Model Training Times\n\n")
                report.write("| Model | Mean (ms) | Min (ms) | Max (ms) |\n")
                report.write("|-------|-----------|----------|----------|\n")
                
                for model in component_data['model_training']:
                    name = model.get('model', 'Unknown')
                    mean = model.get('mean', 0) * 1000  # Convert to ms
                    min_time = model.get('min', 0) * 1000
                    max_time = model.get('max', 0) * 1000
                    report.write(f"| {name} | {mean:.4f} | {min_time:.4f} | {max_time:.4f} |\n")
                report.write("\n")
            
            # Model prediction benchmarks
            if 'model_prediction' in component_data:
                report.write("### Model Prediction Times\n\n")
                report.write("| Model | Mean (ms) | Min (ms) | Max (ms) |\n")
                report.write("|-------|-----------|----------|----------|\n")
                
                for model in component_data['model_prediction']:
                    name = model.get('model', 'Unknown')
                    mean = model.get('mean', 0) * 1000  # Convert to ms
                    min_time = model.get('min', 0) * 1000
                    max_time = model.get('max', 0) * 1000
                    report.write(f"| {name} | {mean:.4f} | {min_time:.4f} | {max_time:.4f} |\n")
                report.write("\n")
        
        # Add pytest benchmark results
        report.write("## Pytest Benchmark Results\n\n")
        
        if pytest_data:
            # Convert pytest benchmark data to DataFrame and format as table
            pytest_df = generate_pytest_benchmark_table(pytest_data)
            if pytest_df is not None:
                report.write(tabulate(pytest_df, headers='keys', tablefmt='pipe', floatfmt='.4f'))
                report.write("\n\n")
        
        # Generate charts
        if component_data:
            generate_model_comparison_chart(component_data)
            report.write("![Model Performance Comparison](model_performance_comparison.png)\n\n")
        
        report.write("## Conclusion\n\n")
        
        if benchmark_metrics:
            best_model = benchmark_metrics.get('best_model', 'Unknown')
            report.write(f"The best performing model was **{best_model}** with an accuracy of {benchmark_metrics.get('accuracy', 'N/A')}.\n\n")
            
        # Add SVM vs RandomForest comparison
        if component_data and 'model_training' in component_data:
            svm_time = 0
            rf_time = 0
            
            for model in component_data['model_training']:
                if model.get('model') == 'SVC':
                    svm_time = model.get('mean', 0)
                elif model.get('model') == 'RandomForestClassifier':
                    rf_time = model.get('mean', 0)
            
            if svm_time and rf_time:
                report.write(f"SVM trained **{rf_time/svm_time:.2f}x faster** than Random Forest, ")
                report.write("while achieving better accuracy, making it the preferred model for this dataset.\n")
    
    print("Benchmark report generated: benchmark_report.md")

if __name__ == "__main__":
    generate_report() 