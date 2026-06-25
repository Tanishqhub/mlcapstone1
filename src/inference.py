import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processor import clean_data, load_artifact

class ChurnPredictor:
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "model.pkl")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        
        # Load artifacts
        try:
            self.preprocessor = load_artifact(self.preprocessor_path)
            self.model = load_artifact(self.model_path)
            print("Predictor successfully loaded preprocessor and model artifacts.")
        except FileNotFoundError as e:
            print(f"Error loading artifacts: {e}")
            print("Make sure you run 'python src/train.py' first to generate them.")
            self.preprocessor = None
            self.model = None

    def predict_dataframe(self, df: pd.DataFrame):
        """
        Runs predictions on a pandas DataFrame.
        Returns:
            predictions: numpy array of binary predictions (0 or 1)
            probabilities: numpy array of churn probabilities
        """
        if self.preprocessor is None or self.model is None:
            raise ValueError("Model artifacts are not loaded. Train the model first.")
            
        # Clean the input dataframe
        df_cleaned = clean_data(df)
        
        # Transform data
        df_processed = self.preprocessor.transform(df_cleaned)
        
        # Make predictions
        predictions = self.model.predict(df_processed)
        probabilities = self.model.predict_proba(df_processed)[:, 1]
        
        return predictions, probabilities

    def predict_single(self, data_dict: dict):
        """
        Runs prediction on a single record dictionary.
        Returns a dict with churn probability and decision.
        """
        df = pd.DataFrame([data_dict])
        preds, probs = self.predict_dataframe(df)
        
        return {
            "churn_probability": float(probs[0]),
            "churn_prediction": int(preds[0]),
            "churn_status": "Yes" if preds[0] == 1 else "No"
        }
        
    def predict_batch_records(self, records: list):
        """
        Runs prediction on a list of record dictionaries.
        Returns a list of dicts with predictions.
        """
        df = pd.DataFrame(records)
        preds, probs = self.predict_dataframe(df)
        
        results = []
        for i in range(len(records)):
            results.append({
                "churn_probability": float(probs[i]),
                "churn_prediction": int(preds[i]),
                "churn_status": "Yes" if preds[i] == 1 else "No"
            })
        return results
