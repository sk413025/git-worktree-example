import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
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
    """Train the machine learning model"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Base model - Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    
    # Print results
    print(f"Model training completed in {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics for comparison
    metrics = {
        'accuracy': accuracy,
        'training_time': training_time
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return metrics

if __name__ == "__main__":
    print("Training baseline model...")
    train_model() 