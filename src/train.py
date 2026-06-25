import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processor import (
    load_and_preprocess_data, get_preprocessor, save_artifact, clean_data
)

def train_model():
    data_path = os.path.join("data", "customer_churn.csv")
    model_dir = "models"
    
    print("--- Step 1: Loading and Preprocessing Data ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Initialize and fit the preprocessing pipeline
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    save_artifact(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
    
    print(f"Preprocessed train shape: {X_train_processed.shape}")
    print(f"Preprocessed test shape: {X_test_processed.shape}")
    
    print("\n--- Step 2: Model Selection & Baseline Training ---")
    
    # 1. Random Forest Baseline
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_processed, y_train)
    rf_preds = rf_model.predict(X_test_processed)
    rf_probs = rf_model.predict_proba(X_test_processed)[:, 1]
    
    # 2. XGBoost Baseline
    xgb_model = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
    xgb_model.fit(X_train_processed, y_train)
    xgb_preds = xgb_model.predict(X_test_processed)
    xgb_probs = xgb_model.predict_proba(X_test_processed)[:, 1]
    
    print("Baseline Random Forest Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, rf_preds):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, rf_probs):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, rf_preds):.4f}")
    
    print("Baseline XGBoost Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, xgb_preds):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, xgb_probs):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, xgb_preds):.4f}")
    
    # Select best model dynamically
    rf_f1 = f1_score(y_test, rf_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    if xgb_f1 >= rf_f1:
        print("\nSelecting XGBoost Classifier for hyperparameter tuning...")
        model_to_tune = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0]
        }
        is_xgb = True
    else:
        print("\nSelecting Random Forest Classifier for hyperparameter tuning...")
        model_to_tune = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10]
        }
        is_xgb = False
        
    print("\n--- Step 3: Hyperparameter Tuning ---")
    grid_search = GridSearchCV(
        estimator=model_to_tune,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_processed, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    print("\n--- Step 4: Model Evaluation ---")
    final_preds = best_model.predict(X_test_processed)
    final_probs = best_model.predict_proba(X_test_processed)[:, 1]
    
    # Save final model
    save_artifact(best_model, os.path.join(model_dir, "model.pkl"))
    
    print("\nFinal Test Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, final_preds):.4f}")
    print(f"Precision: {precision_score(y_test, final_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, final_preds):.4f}")
    print(f"F1-Score:  {f1_score(y_test, final_preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, final_probs):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_preds))

if __name__ == "__main__":
    train_model()
