import unittest
import numpy as np
import os
import json
import pickle
from train import load_data, train_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

class TestModelPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run once before all tests to prepare data"""
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_data()
        
        # Load model if exists, otherwise train
        if os.path.exists('best_model_ensemble.pkl'):
            with open('best_model_ensemble.pkl', 'rb') as f:
                cls.model_data = pickle.load(f)
                cls.model = cls.model_data['model']
                cls.feature_pipeline = cls.model_data['feature_pipeline']
        else:
            # If model not found, we'll train it in the test that needs it
            cls.model_data = None
            
    def test_data_dimensions(self):
        """Test that the data has the expected dimensions"""
        self.assertEqual(self.X_train.shape[1], 10, "Training data should have 10 features")
        self.assertTrue(len(self.y_train) > 0, "Training labels should not be empty")
        
    def test_data_balance(self):
        """Test that the data is reasonably balanced"""
        class_counts = np.bincount(self.y_train)
        # Check that no class has less than 30% of the samples
        min_ratio = min(class_counts) / len(self.y_train)
        self.assertGreater(min_ratio, 0.3, f"Class imbalance detected: {min_ratio:.2f} ratio for minority class")
    
    def test_model_accuracy(self):
        """Test that the model achieves minimum accuracy threshold"""
        if self.model_data is None:
            metrics = train_model()
            self.assertGreaterEqual(metrics['accuracy'], 0.9, "Model accuracy should be at least 90%")
        else:
            # Transform data
            X_test_transformed = self.feature_pipeline.transform(self.X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_transformed)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            self.assertGreaterEqual(accuracy, 0.9, f"Model accuracy should be at least 90%, got {accuracy:.4f}")
    
    def test_model_f1_score(self):
        """Test that the model achieves minimum F1 score threshold"""
        if self.model_data is None:
            # Skip this test if we don't have a model loaded
            self.skipTest("No model available to test F1 score")
        else:
            # Transform data
            X_test_transformed = self.feature_pipeline.transform(self.X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_transformed)
            
            # Calculate F1 score
            f1 = f1_score(self.y_test, y_pred)
            self.assertGreaterEqual(f1, 0.9, f"Model F1 score should be at least 90%, got {f1:.4f}")
    
    def test_feature_engineering(self):
        """Test the feature engineering pipeline"""
        if self.model_data is None:
            self.skipTest("No model pipeline available to test")
        else:
            # Check feature count before and after transformation
            X_train_transformed = self.feature_pipeline.transform(self.X_train)
            self.assertEqual(self.X_train.shape[1], 10, "Original data should have 10 features")
            self.assertEqual(X_train_transformed.shape[1], 30, "Transformed data should have 30 features")

if __name__ == "__main__":
    unittest.main() 