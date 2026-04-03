import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="HeartSense", layout="centered")

# -------------------------
# Load files
# -------------------------
model = joblib.load("heart_best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns(2).pkl")
threshold = joblib.load("best_threshold.pkl")

anfis_models = [
    joblib.load("anfis_model_0.pkl"),
    joblib.load("anfis_model_1.pkl"),
    joblib.load("anfis_model_2.pkl")
]

# -------------------------
# UI
# -------------------------
st.title("❤️ HeartSensse: Heart Disease Prediction System")

st.sidebar.header("Patient Input")

def user_input():
    data = {}
    for col in columns:
        data[col] = st.sidebar.number_input(col, value=0.0)
    return pd.DataFrame([data])

input_df = user_input()

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):

    X = input_df[columns]
    X_scaled = scaler.transform(X)

    # Main model
    prob = model.predict_proba(X_scaled)[0][1]

    # ANFIS
    anfis_preds = [m.predict(X_scaled)[0] for m in anfis_models]
    anfis_risk = np.mean(anfis_preds)

    # IVCFS
    lower = max(0, anfis_risk - 0.1)
    upper = min(1, anfis_risk + 0.1)
    confidence = (1 - (upper - lower)) * 100

    # Risk classification
    if anfis_risk < threshold - 0.1:
        risk = "🟢 Low"
    elif anfis_risk > threshold + 0.1:
        risk = "🔴 High"
    else:
        risk = "🟡 Moderate"

    # Output
    st.subheader("Results")

    st.write(f"**Model Risk:** {prob:.3f}")
    st.write(f"**ANFIS Risk:** {anfis_risk:.3f}")
    st.write(f"**Interval:** [{lower:.3f}, {upper:.3f}]")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.success(f"Final Risk Level: {risk}")