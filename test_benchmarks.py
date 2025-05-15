import pytest
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from train import load_data

# Load test data once for all benchmarks
@pytest.fixture(scope="session")
def data():
    X_train, X_test, y_train, y_test = load_data()
    return X_train, X_test, y_train, y_test

# Model fixture to load the trained model
@pytest.fixture(scope="session")
def model():
    with open('best_model_ensemble.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def test_feature_engineering_performance(benchmark, data):
    """Benchmark the feature engineering pipeline"""
    X_train, _, y_train, _ = data
    
    def run_feature_engineering():
        feature_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('select_features', SelectKBest(f_classif, k=30))
        ])
        return feature_pipeline.fit_transform(X_train, y_train)
    
    # Run the benchmark
    result = benchmark(run_feature_engineering)
    
    # Basic validity check
    assert result.shape[1] == 30, "Feature engineering should produce 30 features"

def test_model_predict_performance(benchmark, data, model):
    """Benchmark the prediction speed of the trained model"""
    _, X_test, _, _ = data
    trained_model = model['model']
    feature_pipeline = model['feature_pipeline']
    
    # Transform test data
    X_test_transformed = feature_pipeline.transform(X_test)
    
    # Benchmark prediction
    predictions = benchmark(lambda: trained_model.predict(X_test_transformed))
    
    # Check that predictions have the right shape
    assert len(predictions) == X_test.shape[0], "Predictions count should match test samples"

def test_full_pipeline_performance(benchmark, data):
    """Benchmark the full classification pipeline including transformation and prediction"""
    X_train, X_test, y_train, y_test = data
    
    def run_full_pipeline():
        # Feature engineering
        feature_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('select_features', SelectKBest(f_classif, k=30))
        ])
        
        # Transform data
        X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        X_test_transformed = feature_pipeline.transform(X_test)
        
        # Train model (use the best model from our previous experiments)
        svm = SVC(kernel='linear', C=10.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train_transformed, y_train)
        
        # Predict
        return svm.predict(X_test_transformed)
    
    # Run the benchmark
    predictions = benchmark(run_full_pipeline)
    
    # Check predictions
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) == X_test.shape[0], "Predictions count should match test samples"

def test_svm_training(benchmark, data):
    """Test SVM training performance"""
    X_train, _, y_train, _ = data
    
    # Create feature engineered data
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('select_features', SelectKBest(f_classif, k=30))
    ])
    X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
    
    # Test SVM training
    def train_svm():
        svm = SVC(kernel='linear', C=10.0, gamma='scale', random_state=42)
        svm.fit(X_train_transformed, y_train)
        return svm
    
    benchmark(train_svm)

def test_rf_training(benchmark, data):
    """Test RandomForest training performance"""
    X_train, _, y_train, _ = data
    
    # Create feature engineered data
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('select_features', SelectKBest(f_classif, k=30))
    ])
    X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
    
    # Test RandomForest training
    def train_rf():
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_transformed, y_train)
        return rf
    
    benchmark(train_rf) 