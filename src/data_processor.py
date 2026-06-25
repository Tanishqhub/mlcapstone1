import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Define feature groups
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET_COL = "Churn"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data.
    Coerces TotalCharges to float and handles empty spaces as NaN.
    """
    df = df.copy()
    
    # Coerce TotalCharges to numeric (handles empty spaces as NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        
    return df

def get_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer for preprocessing.
    Uses SimpleImputer and StandardScaler for numerical columns.
    Uses SimpleImputer and OneHotEncoder for categorical columns.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS)
        ],
        remainder="drop"
    )
    
    return preprocessor

def load_and_preprocess_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads data, cleans it, splits features/target, performs train/test split.
    Returns: X_train, X_test, y_train, y_test
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # Process target label
    if TARGET_COL in df.columns:
        # Convert Yes/No to 1/0
        y = df[TARGET_COL].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        X = df.drop(columns=[TARGET_COL, "customerID"], errors="ignore")
    else:
        y = None
        X = df.drop(columns=["customerID"], errors="ignore")
        
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    else:
        return X

def save_artifact(obj, filepath: str):
    """Saves a pickle artifact (model or preprocessor) to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artifact saved to: {filepath}")

def load_artifact(filepath: str):
    """Loads a pickle artifact from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at: {filepath}")
    return joblib.load(filepath)
