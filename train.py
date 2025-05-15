import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import time
import pickle
import json

def load_data():
    """Load sample data (for demonstration)"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    """Train combined experiment with feature engineering, hyperparameter tuning and model selection"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Step 1: Create feature engineering pipeline
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('poly', PolynomialFeatures(degree=2, interaction_only=True)),  # Add interaction features
    ])
    
    # Apply feature engineering
    print("Applying feature engineering...")
    X_train_transformed = feature_pipeline.fit_transform(X_train)
    X_test_transformed = feature_pipeline.transform(X_test)
    
    print(f"Original feature count: {X_train.shape[1]}")
    print(f"New feature count after engineering: {X_train_transformed.shape[1]}")
    
    # Step 2: Define models to evaluate with their hyperparameter grids
    model_params = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    # Dictionary to store best model from each type
    best_models = {}
    model_performance = {}
    
    # Step 3: For each model type, perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning for each model type...")
    for model_name, config in model_params.items():
        print(f"\nTuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train_transformed, y_train)
        
        # Store best model from this type
        best_models[model_name] = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        print(f"{model_name} best CV accuracy: {best_score:.4f}")
        print(f"{model_name} best parameters: {best_params}")
        
        # Record performance
        model_performance[model_name] = {
            'cv_accuracy': best_score,
            'best_params': best_params
        }
    
    # Step 4: Find best overall model
    best_model_name = max(model_performance, key=lambda x: model_performance[x]['cv_accuracy'])
    best_model = best_models[best_model_name]
    
    print(f"\nBest overall model: {best_model_name}")
    print(f"Best overall CV accuracy: {model_performance[best_model_name]['cv_accuracy']:.4f}")
    
    # Step 5: Final evaluation on test set
    y_pred = best_model.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save best model and pipeline
    with open('best_model_ensemble.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_pipeline': feature_pipeline
        }, f)
    
    training_time = time.time() - start_time
    
    # Save metrics for comparison
    metrics = {
        'accuracy': test_accuracy,
        'training_time': training_time,
        'best_model': best_model_name,
        'best_params': model_performance[best_model_name]['best_params'],
        'original_features': X_train.shape[1],
        'engineered_features': X_train_transformed.shape[1],
        'model_performances': {k: {'cv_accuracy': float(v['cv_accuracy'])} for k, v in model_performance.items()}
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return metrics

if __name__ == "__main__":
    print("Starting ensemble experiment with feature engineering, hyperparameter tuning and model selection...")
    train_model() 