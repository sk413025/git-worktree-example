import timeit
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from train import load_data

def setup_data():
    """Prepare data for benchmarking"""
    X_train, X_test, y_train, y_test = load_data()
    return X_train, X_test, y_train, y_test

def benchmark_feature_engineering(X_train, y_train):
    """Benchmark the feature engineering pipeline"""
    setup_code = """
from __main__ import X_train, y_train
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
"""
    
    test_code = """
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('select_features', SelectKBest(f_classif, k=30))
])
X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
"""
    
    times = timeit.repeat(stmt=test_code, setup=setup_code, repeat=5, number=10)
    return {
        "mean": np.mean(times) / 10,  # Average time per execution
        "min": np.min(times) / 10,
        "max": np.max(times) / 10,
        "values": [t / 10 for t in times]  # Time per execution
    }

def benchmark_model_training(model_class, X_train, y_train, **model_params):
    """Benchmark training of a specific model"""
    model_name = model_class.__name__
    
    # Create model instance with parameters
    model_instance = model_class(**model_params)
    
    def train_model():
        model_instance.fit(X_train, y_train)
    
    # Run benchmark
    times = []
    for _ in range(5):
        start_time = timeit.default_timer()
        train_model()
        times.append(timeit.default_timer() - start_time)
    
    return {
        "model": model_name,
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "values": times
    }

def benchmark_prediction_speed(model_class, X_train, X_test, y_train, **model_params):
    """Benchmark prediction speed of a trained model"""
    model_name = model_class.__name__
    
    # Train model first
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # Benchmark prediction
    def predict():
        return model.predict(X_test)
    
    # Run benchmark
    times = timeit.repeat(predict, repeat=10, number=100)
    
    return {
        "model": model_name,
        "mean": np.mean(times) / 100,  # Average time per execution
        "min": np.min(times) / 100,
        "max": np.max(times) / 100,
        "values": [t / 100 for t in times]  # Time per execution
    }

if __name__ == "__main__":
    print("Running focused microbenchmarks...")
    
    # Load data
    X_train, X_test, y_train, y_test = setup_data()
    
    # Create feature engineered data for model benchmarks
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('select_features', SelectKBest(f_classif, k=30))
    ])
    X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = feature_pipeline.transform(X_test)
    
    benchmarks = {
        "feature_engineering": benchmark_feature_engineering(X_train, y_train)
    }
    
    # Benchmark model training
    model_configs = [
        (RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}),
        (GradientBoostingClassifier, {'n_estimators': 50, 'learning_rate': 0.1, 'random_state': 42}),
        (SVC, {'kernel': 'linear', 'C': 10.0, 'gamma': 'scale', 'random_state': 42}),
        (MLPClassifier, {'hidden_layer_sizes': (50, 25), 'activation': 'tanh', 'alpha': 0.01, 'random_state': 42})
    ]
    
    training_benchmarks = []
    prediction_benchmarks = []
    
    for model_class, params in model_configs:
        print(f"Benchmarking {model_class.__name__}...")
        
        # Benchmark training
        training_result = benchmark_model_training(model_class, X_train_transformed, y_train, **params)
        training_benchmarks.append(training_result)
        
        # Benchmark prediction
        prediction_result = benchmark_prediction_speed(model_class, X_train_transformed, X_test_transformed, y_train, **params)
        prediction_benchmarks.append(prediction_result)
    
    benchmarks["model_training"] = training_benchmarks
    benchmarks["model_prediction"] = prediction_benchmarks
    
    # Save benchmark results
    with open('component_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    print("\nMicrobenchmarks complete.")
    print("Results saved to component_benchmarks.json") 