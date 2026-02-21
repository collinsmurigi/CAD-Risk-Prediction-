import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import os

class CADMLModel:
    """
    Machine Learning Model for CAD Prediction using UCI Heart Disease Dataset
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal'
        ]
        self.performance_metrics = {}
        
    def load_and_preprocess_data(self, data_path='data/heart.csv'):
        """
        Load and preprocess the UCI Heart Disease dataset
        """
        # Column names for the dataset
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'target'
        ]
        
        # Load dataset
        df = pd.read_csv(data_path, names=column_names, na_values='?')
        
        # Handle missing values
        df = df.replace('?', np.nan)
        for col in ['ca', 'thal']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for training
        """
        X = df[self.feature_names]
        y = df['target']
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_names)
        
        return X, y
    
    def train(self, X, y):
        """
        Train the selected model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        self.performance_metrics['cv_mean'] = cv_scores.mean()
        self.performance_metrics['cv_std'] = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.performance_metrics['feature_importance'] = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
        elif hasattr(self.model, 'coef_'):
            self.performance_metrics['feature_importance'] = dict(
                zip(self.feature_names, np.abs(self.model.coef_[0]))
            )
        
        return self.performance_metrics
    
    def predict(self, patient_data):
        """
        Predict CAD risk for a single patient
        """
        # Create feature vector
        features = np.array([[
            patient_data.get('age', 0),
            patient_data.get('sex', 0),
            patient_data.get('cp', 0),
            patient_data.get('trestbps', 0),
            patient_data.get('chol', 0),
            patient_data.get('fbs', 0),
            patient_data.get('restecg', 0),
            patient_data.get('thalach', 0),
            patient_data.get('exang', 0),
            patient_data.get('oldpeak', 0),
            patient_data.get('slope', 0),
            patient_data.get('ca', 0),
            patient_data.get('thal', 0)
        ]])
        
        # Handle missing values
        features = self.imputer.transform(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Categorize risk
        if probability < 0.2:
            category = "Low"
        elif probability < 0.5:
            category = "Medium"
        else:
            category = "High"
        
        return {
            'probability': round(probability * 100, 2),
            'category': category,
            'raw_probability': probability
        }
    
    def save_model(self, filepath='models/cad_model.pkl'):
        """
        Save trained model to disk
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/cad_model.pkl'):
        """
        Load trained model from disk
        """
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.feature_names = model_data['feature_names']
            self.performance_metrics = model_data['performance_metrics']
            self.model_type = model_data['model_type']
            print(f"Model loaded from {filepath}")
            return True
        return False

# Training script
def train_and_save_model():
    """
    Train the model and save it
    """
    # Download dataset if not exists
    data_path = 'data/heart.csv'
    if not os.path.exists(data_path):
        print("Please download the Cleveland Heart Disease dataset from UCI")
        print("URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
        print("Save it as 'data/heart.csv'")
        return
    
    # Initialize model
    ml_model = CADMLModel(model_type='random_forest')
    
    # Load and preprocess data
    print("Loading dataset...")
    df = ml_model.load_and_preprocess_data(data_path)
    
    # Prepare features
    print("Preparing features...")
    X, y = ml_model.prepare_features(df)
    
    # Train model
    print("Training model...")
    metrics = ml_model.train(X, y)
    
    # Print performance
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Cross-validation: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
    
    if 'feature_importance' in metrics:
        print("\nTop 5 Important Features:")
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.3f}")
    
    # Save model
    ml_model.save_model()
    
    return ml_model

if __name__ == "__main__":
    train_and_save_model()