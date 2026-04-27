
import streamlit as st
import pandas as pd
import joblib
import os

# Load model
try:
    data = joblib.load("hospital_model.pkl")
    model = data["model"]
    columns = data["columns"]
except FileNotFoundError:
    st.error("Model file 'hospital_model.pkl' not found. Please run the training notebook first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("Hospital Appointment Prediction")
st.write("Predict whether a patient will show up for their appointment")

# Create input form
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    age = st.number_input("Age", 0, 100, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    scholarship = st.selectbox("Scholarship", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    alcoholism = st.selectbox("Alcoholism", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    handicap = st.selectbox("Handicap", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    sms_received = st.selectbox("SMS Received", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Hypertension": [hypertension],
        "Diabetes": [diabetes],
        "Scholarship": [scholarship],
        "Alcoholism": [alcoholism],
        "Handicap": [handicap],
        "SMS_received": [sms_received]
    })
    
    # Match columns with training data
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    result = "Yes (Patient will NOT show up)" if prediction == 1 else "No (Patient will show up)"
    
    st.success(f"## Prediction: {result}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Will NOT Show Up", f"{probability[1]:.1%}")
    with col2:
        st.metric("Will Show Up", f"{probability[0]:.1%}")
