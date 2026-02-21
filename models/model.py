import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class AdvancedCADModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            # Create a simple logistic regression model
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42)
        
        self.scaler = StandardScaler()
        self.feature_names = ['age', 'total_cholesterol', 'hdl_cholesterol', 
                             'systolic_bp', 'smoking_current', 'diabetes', 
                             'family_history']
    
    def preprocess(self, data):
        """Preprocess input data"""
        # Convert categorical variables
        features = {
            'age': data['age'],
            'total_cholesterol': data['total_cholesterol'],
            'hdl_cholesterol': data['hdl_cholesterol'],
            'systolic_bp': data['systolic_bp'],
            'smoking_current': 1 if data['smoking_status'] == 'current' else 0,
            'diabetes': 1 if data['diabetes_status'] else 0,
            'family_history': 1 if data['family_history_cad'] else 0
        }
        
        # Create feature array
        X = np.array([list(features.values())])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def predict(self, data):
        """Make prediction"""
        X = self.preprocess(data)
        
        # Get probability
        proba = self.model.predict_proba(X)[0][1]
        
        # Categorize
        if proba < 0.1:
            category = "Low"
        elif proba < 0.2:
            category = "Medium"
        else:
            category = "High"
        
        return {
            'probability': round(proba * 100, 2),
            'category': category,
            'confidence_interval': {
                'lower': max(0, round((proba - 0.05) * 100, 2)),
                'upper': min(100, round((proba + 0.05) * 100, 2))
            }
        }