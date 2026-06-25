import streamlit as st
import pandas as pd
import numpy as np
import httpx
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Portal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium dashboard look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    }
    
    /* Header/Title styles */
    h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Card Style */
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    /* Metrics panel */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Churn Status styling */
    .status-low {
        color: #10b981;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .status-med {
        color: #f59e0b;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .status-high {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    /* Custom button behavior */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Input label styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>🔮 Customer Churn Intelligence Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A production-grade Predictive AI Platform to analyze subscriber retention and prevent churn.</p>", unsafe_allow_html=True)

# FastAPI URL
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/clouds/200/database.png", width=120)
st.sidebar.markdown("<h2 style='color:#38bdf8;'>Navigation</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Go to", ["Single Customer Predictor", "Batch CSV Predictor", "System Health Check"])

# Helper to query FastAPI
def check_api_health():
    try:
        response = httpx.get(f"{API_URL}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False

# Single Customer Predictor Mode
if app_mode == "Single Customer Predictor":
    st.subheader("Predict Retention Risk for an Individual Subscriber")
    
    # Check health and display connection warning
    api_online = check_api_health()
    if not api_online:
        st.warning(f"⚠️ FastAPI Backend ({API_URL}) is currently offline. Please run the FastAPI app first (`uvicorn app:app`). Attempting to predict will fail.")
    
    # Form layout inside custom card container
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
        st.markdown("### Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])

    with col2:
        st.markdown("### Core Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        st.markdown("### Internet Add-ons")
        online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection Add-on", ["No", "Yes", "No internet service"])
        
    with col3:
        st.markdown("### Entertainment & Support")
        tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.markdown("### Financials")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
        
        # Calculate standard TotalCharges based on monthly charges & tenure
        calculated_total = round(monthly_charges * tenure, 2)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8600.0, value=calculated_total)

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("Evaluate Churn Risk"):
        # Format payload
        payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
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
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": str(total_charges)
        }
        
        with st.spinner("Scoring customer profile..."):
            try:
                response = httpx.post(f"{API_URL}/predict", json=payload, timeout=5.0)
                if response.status_code == 200:
                    res_data = response.json()
                    prob = res_data["churn_probability"]
                    churn_status = res_data["churn_status"]
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Retention Analysis Report")
                    
                    r_col1, r_col2 = st.columns([1, 2])
                    
                    with r_col1:
                        # Risk status formatting
                        if prob < 0.3:
                            risk_class = "status-low"
                            risk_lbl = "LOW RISK"
                        elif prob < 0.6:
                            risk_class = "status-med"
                            risk_lbl = "MEDIUM RISK"
                        else:
                            risk_class = "status-high"
                            risk_lbl = "HIGH RISK"
                            
                        st.markdown(f"Churn Classification: <span class='{risk_class}'>{risk_lbl}</span>", unsafe_allow_html=True)
                        st.metric(label="Churn Probability Score", value=f"{prob * 100:.1f}%")
                        st.progress(prob)
                        
                    with r_col2:
                        st.markdown("### Key Observations & Action Items")
                        if prob >= 0.6:
                            st.error("🚨 **Critical Churn Risk Identified.**")
                            reasons = []
                            if contract == "Month-to-month":
                                reasons.append("Customer is on a flexible Month-to-month contract.")
                            if internet_service == "Fiber optic":
                                reasons.append("Fiber optic service has a statistically higher churn trend.")
                            if tech_support == "No":
                                reasons.append("No technical support subscription adds to friction.")
                            if tenure < 12:
                                reasons.append("New customer in high-friction initial tenure period.")
                                
                            for r in reasons:
                                st.write(f"- {r}")
                            st.write("**Recommendation:** Proactively reach out with long-term retention discounts (e.g. 1-year contract conversion) or free Tech Support add-ons.")
                        elif prob >= 0.3:
                            st.warning("⚠️ **Moderate Churn Risk.**")
                            st.write("Subscriber exhibits moderate churn probability. Suggest soft touchpoints (feedback survey, newsletter, feature highlight of existing addons).")
                        else:
                            st.success("✅ **Healthy Customer Profile.**")
                            st.write("Subscriber has stable tenure patterns. Continue regular service, target for potential upgrades or premium service offerings.")
                            
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Error from FastAPI backend: {response.text}")
            except Exception as e:
                st.error(f"Failed to communicate with FastAPI at {API_URL}. Details: {str(e)}")

# Batch CSV Predictor Mode
elif app_mode == "Batch CSV Predictor":
    st.subheader("Process Batch Predictions via CSV File Upload")
    
    st.write("Upload a CSV file containing subscriber records. Ensure the CSV has headers matching the churn dataset features.")
    
    # Show sample format expander
    with st.expander("Show expected CSV format structure"):
        sample_df = pd.DataFrame([{
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.80, "TotalCharges": 849.60
        }])
        st.dataframe(sample_df)
        
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load file
        df_input = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded file containing {len(df_input)} customer records.")
        st.dataframe(df_input.head(5))
        
        if st.button("Generate Batch Predictions"):
            # Check backend
            if not check_api_health():
                st.error("FastAPI Backend is offline. Cannot proceed with integration request.")
            else:
                # Prepare payload
                df_input_clean = df_input.copy()
                df_input_clean["SeniorCitizen"] = df_input_clean["SeniorCitizen"].fillna(0).astype(int)
                df_input_clean["tenure"] = df_input_clean["tenure"].fillna(1).astype(int)
                df_input_clean["MonthlyCharges"] = df_input_clean["MonthlyCharges"].fillna(20.0).astype(float)
                df_input_clean["TotalCharges"] = df_input_clean["TotalCharges"].fillna("0").astype(str)
                
                records = df_input_clean.to_dict(orient="records")
                
                with st.spinner("Communicating with prediction service..."):
                    try:
                        response = httpx.post(f"{API_URL}/predict_batch", json=records, timeout=30.0)
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Merge results back into DataFrame
                            probs = [r["churn_probability"] for r in results]
                            preds = [r["churn_status"] for r in results]
                            
                            df_output = df_input.copy()
                            df_output["Churn_Probability"] = probs
                            df_output["Churn_Prediction"] = preds
                            
                            st.markdown("### Churn Analysis Results")
                            
                            # Show summary stats
                            churn_pct = (df_output["Churn_Prediction"] == "Yes").mean() * 100
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total Records Processed", len(df_output))
                            c2.metric("Predicted Churn Cases", sum(df_output["Churn_Prediction"] == "Yes"))
                            c3.metric("Expected Churn Rate", f"{churn_pct:.1f}%")
                            
                            # Visualization
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            col_chart, col_table = st.columns([1, 1])
                            
                            with col_chart:
                                st.markdown("#### Risk Distribution")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.histplot(df_output["Churn_Probability"], bins=15, kde=True, ax=ax, color="#4f46e5")
                                ax.set_xlabel("Churn Probability")
                                ax.set_ylabel("Count")
                                ax.set_title("Distribution of Churn Probabilities")
                                # Transparent/Dark theme styling for chart
                                fig.patch.set_facecolor('#1e293b')
                                ax.set_facecolor('#0f172a')
                                ax.xaxis.label.set_color('#cbd5e1')
                                ax.yaxis.label.set_color('#cbd5e1')
                                ax.title.set_color('#f1f5f9')
                                ax.tick_params(colors='#cbd5e1')
                                st.pyplot(fig)
                                
                            with col_table:
                                st.markdown("#### High Risk Customers")
                                high_risk = df_output[df_output["Churn_Probability"] >= 0.6].sort_values("Churn_Probability", ascending=False)
                                if len(high_risk) > 0:
                                    st.dataframe(high_risk[["tenure", "Contract", "MonthlyCharges", "Churn_Probability"]].head(10))
                                else:
                                    st.info("No customers predicted to have a high risk (>=60%) of churn.")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Full table and download
                            st.markdown("#### Full Predicted Cohort")
                            st.dataframe(df_output)
                            
                            # Export CSV button
                            csv_data = df_output.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Prediction Results CSV",
                                data=csv_data,
                                file_name="churn_predictions_export.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Error from FastAPI backend: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with FastAPI at {API_URL}. Details: {str(e)}")

# System Health Check Mode
elif app_mode == "System Health Check":
    st.subheader("System Diagnostics")
    
    st.write("Verifying integration between Streamlit UI frontend and FastAPI prediction API backend.")
    
    api_online = check_api_health()
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if api_online:
        st.success("🟢 **FastAPI Service is Online and Healthy**")
        st.write(f"Connected to backend service at: `{API_URL}`")
        
        try:
            res = httpx.get(f"{API_URL}/health")
            st.json(res.json())
        except Exception:
            pass
    else:
        st.error("🔴 **FastAPI Service is Offline**")
        st.write(f"Unable to connect to `{API_URL}`.")
        st.write("To start the backend:")
        st.code("uvicorn app:app --port 8000 --reload")
        
    st.write("---")
    st.markdown("#### Directory Check:")
    st.write(f"Active workspace path: `{os.getcwd()}`")
    st.write(f"Model File (model.pkl) exists: `{os.path.exists(os.path.join('models', 'model.pkl'))}`")
    st.write(f"Preprocessor (preprocessor.pkl) exists: `{os.path.exists(os.path.join('models', 'preprocessor.pkl'))}`")
    st.write(f"Dataset (customer_churn.csv) exists: `{os.path.exists(os.path.join('data', 'customer_churn.csv'))}`")
    st.markdown("</div>", unsafe_allow_html=True)
