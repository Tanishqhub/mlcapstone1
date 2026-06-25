# Customer Churn Intelligence Portal

This is an intermediate-grade Machine Learning capstone project designed to predict and analyze subscriber retention (customer churn). 
The project includes a complete ML pipeline starting with exploratory data analysis in a Jupyter Notebook and ending with a production-grade FastAPI serving backend integrated with a visual Streamlit web dashboard.

---

## Project Architecture

```
project1/
├── data/
│   ├── generate_data.py          # Script to generate synthetic churn data
│   └── customer_churn.csv        # Generated dataset
├── models/
│   ├── model.pkl                 # Trained ML model (XGBoost Classifier)
│   └── preprocessor.pkl          # Fitted preprocessor pipeline (scaling, encoding)
├── notebooks/
│   └── eda_and_prototyping.ipynb # Notebook covering steps 1-10 (EDA to Model Saving)
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Data preprocessing and pipeline definition
│   ├── train.py                  # Production training & hyperparameter tuning script
│   └── inference.py              # Production prediction engine
├── app.py                        # FastAPI Backend API (serving predictions)
├── ui.py                         # Streamlit Frontend UI (user interface)
├── requirements.txt              # Project dependencies
└── README.md                     # Project Documentation
```

---

## 1. Problem Statement
**Objective**: Build a predictive model that identifies which subscribers are at risk of leaving a telecom subscription.
**Value Proposition**: Customer acquisition costs are significantly higher than retention costs. By predicting high-risk churn customers, the business can deploy targeted interventions (e.g., tailored discounts, service contract updates) to retain users and save revenue.

---

## 2. Dataset
The project utilizes a synthetically generated dataset with 2,500 subscriber profiles. It mirrors the exact schema of standard industry benchmarks (like the Telco Customer Churn dataset on Kaggle) and includes:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Billing**: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes / No)

---

## 3. Key Findings & Summary
1. **Contract Types**: Customers with Month-to-month contracts have the highest likelihood of churning (over 45%). One-year and Two-year contracts are extremely stable.
2. **Tenure Effect**: Lower tenure (new subscribers under 6 months) exhibits the highest risk of churn, highlighting a need for a smooth onboarding experience.
3. **Payment Methods**: Electronic Check payments correlate with a high risk of churn, while automatic payment methods (Credit Card or Bank Transfer) show very high retention rates.
4. **Service Quality**: Subscribers using Fiber Optic internet services show higher churn rates than DSL users, pointing to potential customer service or pricing friction in premium plans.

---

## 4. How to Run the Project

### Step 1: Activate Virtual Environment and Install Dependencies
Activate the virtual environment:
```powershell
# Windows PowerShell
venv\Scripts\activate
```
Install requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Generate the Dataset
Create the raw dataset programmatically:
```bash
python data/generate_data.py
```

### Step 3: Run Jupyter Prototyping
Start the Jupyter environment and run the notebook:
```bash
jupyter notebook notebooks/eda_and_prototyping.ipynb
```

### Step 4: Run Production Model Training
Train, tune, and evaluate the final model:
```bash
python src/train.py
```

### Step 5: Start the FastAPI Backend
Start the prediction server:
```bash
uvicorn app:app --port 8000 --reload
```
You can view the interactive Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 6: Start the Streamlit Dashboard
In a new terminal window (with the venv activated), launch the UI:
```bash
streamlit run ui.py
```
Open the provided local URL (usually [http://localhost:8501](http://localhost:8501)) in your browser.

---

## 5. Integration and Testing
The Streamlit dashboard operates in two modes:
- **Single Customer Predictor**: Form sliders and dropdowns to score a single customer. It hits FastAPI's `POST /predict` endpoint, displaying risk alerts (LOW, MEDIUM, HIGH) and dynamic recommendations.
- **Batch CSV Predictor**: Uploads a CSV list of subscribers, sends them in bulk to FastAPI's `POST /predict_batch`, and renders an interactive dashboard showing predictions, churn ratios, charts, and downloadable prediction outputs.
