import os
import sys
from typing import List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference import ChurnPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service to predict likelihood of customer churn based on subscription and demographic features.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the predictor lazily
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = ChurnPredictor()
    return predictor

# Define Pydantic Schema for inputs
class CustomerInput(BaseModel):
    gender: str = Field(..., description="Gender of the customer (Male, Female)")
    SeniorCitizen: int = Field(..., description="Whether the customer is a senior citizen (1, 0)")
    Partner: str = Field(..., description="Whether the customer has a partner (Yes, No)")
    Dependents: str = Field(..., description="Whether the customer has dependents (Yes, No)")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., description="Whether the customer has a phone service (Yes, No)")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines (Yes, No, No phone service)")
    InternetService: str = Field(..., description="Customer's internet service provider (DSL, Fiber optic, No)")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security (Yes, No, No internet service)")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup (Yes, No, No internet service)")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection (Yes, No, No internet service)")
    TechSupport: str = Field(..., description="Whether the customer has tech support (Yes, No, No internet service)")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV (Yes, No, No internet service)")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies (Yes, No, No internet service)")
    Contract: str = Field(..., description="The contract term of the customer (Month-to-month, One year, Two year)")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing (Yes, No)")
    PaymentMethod: str = Field(..., description="The customer's payment method")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly")
    TotalCharges: Union[str, float] = Field(..., description="The total amount charged to the customer (can be space string ' ' or numeric)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.8,
                "TotalCharges": "849.6"
            }
        }
    }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Customer Churn Prediction API",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    prod = get_predictor()
    if prod.model is None or prod.preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model artifacts are not loaded. Run training script first."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    prod = get_predictor()
    if prod.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Convert Pydantic object to dict
    customer_dict = customer.model_dump()
    try:
        prediction = prod.predict_single(customer_dict)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_churn_batch(customers: List[CustomerInput]):
    prod = get_predictor()
    if prod.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    records = [c.model_dump() for c in customers]
    try:
        predictions = prod.predict_batch_records(records)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
