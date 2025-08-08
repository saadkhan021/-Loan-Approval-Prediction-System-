import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\saadk\Desktop\Week4 intern\models\best_model.pkl")


model = load_model()

# App title and description
st.title(" Loan Approval Prediction")
st.markdown("Fill in the form below to check if your loan is likely to be approved.")

# User input form
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1)
    loan_term = st.selectbox("Loan Amount Term (in months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submitted = st.form_submit_button("Predict")

# Encode user input just like training
def preprocess_input():
    return pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        "Education": 0 if education == "Graduate" else 1,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    }])

# Show prediction
if submitted:
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    result = " Loan Approved" if prediction == 1 else " Loan Not Approved"
    st.success(f"Prediction Result: {result}")
