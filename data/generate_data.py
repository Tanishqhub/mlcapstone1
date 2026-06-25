import os
import numpy as np
import pandas as pd

def generate_customer_churn_data(n_samples=2000, seed=42):
    np.random.seed(seed)
    
    # 1. Customer IDs
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    
    # 2. Demographics
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])
    
    # 3. Tenure and Contract
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], 
        size=n_samples, 
        p=[0.55, 0.20, 0.25]
    )
    
    # Tenure depends on contract
    tenure = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if contract[i] == "Month-to-month":
            tenure[i] = np.random.randint(1, 24)  # Shorter tenure
        elif contract[i] == "One year":
            tenure[i] = np.random.randint(12, 48) # Medium tenure
        else:
            tenure[i] = np.random.randint(24, 73) # Longer tenure
            
    # 4. Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for i in range(n_samples):
        if phone_service[i] == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.42, 0.58]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.45, 0.20])
    
    # Internet sub-services depend on internet service
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for i in range(n_samples):
        if internet_service[i] == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.41, 0.59]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.36, 0.64]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.48, 0.52]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.49, 0.51]))
            
    # 5. Billing and Payments
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # 6. Monthly Charges calculation
    monthly_charges = np.zeros(n_samples)
    for i in range(n_samples):
        cost = 20.0  # Base cost
        if phone_service[i] == "Yes":
            cost += 10.0
            if multiple_lines[i] == "Yes":
                cost += 15.0
        if internet_service[i] == "DSL":
            cost += 30.0
        elif internet_service[i] == "Fiber optic":
            cost += 50.0
            
        if internet_service[i] != "No":
            if online_security[i] == "Yes": cost += 8.0
            if online_backup[i] == "Yes": cost += 8.0
            if device_protection[i] == "Yes": cost += 8.0
            if tech_support[i] == "Yes": cost += 8.0
            if streaming_tv[i] == "Yes": cost += 10.0
            if streaming_movies[i] == "Yes": cost += 10.0
            
        # Add random noise
        cost += np.random.normal(0, 3.0)
        monthly_charges[i] = max(18.0, round(cost, 2))
        
    # 7. Total Charges calculation
    total_charges = []
    for i in range(n_samples):
        total = monthly_charges[i] * tenure[i]
        # Add slight noise
        total += np.random.normal(0, monthly_charges[i] * 0.05)
        total = max(monthly_charges[i], total)
        # Introduce very few missing values in TotalCharges to demonstrate handling missing values in EDA/Feature engineering (around 0.5%)
        if np.random.rand() < 0.005:
            total_charges.append(" ") # Empty string like Kaggle's dataset
        else:
            total_charges.append(f"{total:.2f}")
            
    # 8. Churn calculation based on logic
    churn_prob = np.zeros(n_samples)
    for i in range(n_samples):
        # Base probability
        p = 0.15
        
        # Contract risk
        if contract[i] == "Month-to-month":
            p += 0.35
        elif contract[i] == "One year":
            p += 0.05
            
        # Tenure risk
        if tenure[i] < 6:
            p += 0.20
        elif tenure[i] < 12:
            p += 0.10
        elif tenure[i] > 48:
            p -= 0.15
            
        # Service risk
        if internet_service[i] == "Fiber optic":
            p += 0.10
        if internet_service[i] != "No" and tech_support[i] == "No":
            p += 0.08
        if internet_service[i] != "No" and online_security[i] == "No":
            p += 0.08
            
        # Billing / Demographics risk
        if payment_method[i] == "Electronic check":
            p += 0.12
        if senior_citizen[i] == 1:
            p += 0.05
        if dependents[i] == "Yes":
            p -= 0.05
            
        # Bound probability between 0.01 and 0.99
        p = min(max(p, 0.01), 0.99)
        churn_prob[i] = p
        
    # Determine churn label
    churn = np.where(np.random.rand(n_samples) < churn_prob, "Yes", "No")
    
    # Create DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic Customer Churn dataset...")
    df = generate_customer_churn_data(n_samples=2500)
    
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save CSV
    output_path = os.path.join("data", "customer_churn.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
