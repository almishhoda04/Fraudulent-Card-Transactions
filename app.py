import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
xgb = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# UI Branding
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.markdown("<h1 style='text-align: center; color: #003366;'>💳 Credit Card Fraud Detection </h1>", unsafe_allow_html=True)
st.markdown("#### Enter transaction details below. Try different values to see how predictions change.")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
- Enter values for as many features as you want.
- Unfilled features will be set to zero.
- The transaction amount will be automatically calculated as the sum of absolute feature values.
- Click 'Predict' to see if the transaction is fraudulent.
""")

st.sidebar.markdown("---") 
st.sidebar.markdown(
    "<p style='text-align:center; color:grey;'>Made by Almish with love ❤️</p>",
    unsafe_allow_html=True
)

# Hardcode feature names (V1-V28 + Amount)
feature_names = [f'V{i}' for i in range(1, 29)]
num_features = len(feature_names)

# Input fields in columns for better UI
cols = st.columns(4)
inputs = []
for i, feature in enumerate(feature_names):
    value = cols[i % 4].number_input(feature, value=0.0)
    inputs.append(value)

# Calculate Amount automatically
amount = sum(abs(x) for x in inputs)
st.write(f"**Calculated Transaction Amount:** {amount:.2f}")

inputs.append(amount)  # Amount is last feature

# Prediction
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred_proba = xgb.predict_proba(input_scaled)[0,1]
    pred = xgb.predict(input_scaled)[0]
    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! Probability: {pred_proba:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {pred_proba:.2f}")