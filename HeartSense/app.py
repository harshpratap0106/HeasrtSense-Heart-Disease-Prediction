import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="HeartSense", layout="centered")

st.title("❤️ HeartSense: Heart Disease Prediction System")

# -------------------------
# Load Models (FIXED PATH)
# -------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "heart_best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model_columns (2).pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "best_threshold.pkl"))

# -------------------------
# Sidebar Input UI
# -------------------------
st.sidebar.header("🩺 Enter Patient Details")

def user_input():
    data = {}

    # Age
    data["age"] = st.sidebar.slider("Age", 20, 100, 40)

    # Sex
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    data["sex"] = 1 if sex == "Male" else 0

    # Chest Pain Type
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    data["cp"] = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

    # Resting Blood Pressure
    data["trestbps"] = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)

    # Cholesterol
    data["chol"] = st.sidebar.slider("Cholesterol", 100, 400, 200)

    # Fasting Blood Sugar
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    data["fbs"] = 1 if fbs == "Yes" else 0

    # Rest ECG
    restecg = st.sidebar.selectbox(
        "Resting ECG",
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
    )
    data["restecg"] = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

    # Max Heart Rate
    data["thalach"] = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

    # Exercise Induced Angina
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    data["exang"] = 1 if exang == "Yes" else 0

    # ST Depression
    data["oldpeak"] = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

    # Slope
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    data["slope"] = ["Upsloping", "Flat", "Downsloping"].index(slope)

    # Number of Major Vessels
    data["ca"] = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)

    # Thal
    thal = st.sidebar.selectbox(
        "Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    data["thal"] = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

    return pd.DataFrame([data])

input_df = user_input()

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict"):

    try:
        # Ensure correct column order
        X = input_df[columns]

        # Scale
        X_scaled = scaler.transform(X)

        # Prediction
        prob = model.predict_proba(X_scaled)[0][1]

        # TEMP FIX: Use model instead of ANFIS
        risk_score = prob

        # Risk Levels
        if risk_score < threshold - 0.1:
            risk = "🟢 Low Risk"
        elif risk_score > threshold + 0.1:
            risk = "🔴 High Risk"
        else:
            risk = "🟡 Moderate Risk"

        # Confidence
        confidence = (1 - abs(risk_score - threshold)) * 100

        # -------------------------
        # Output
        # -------------------------
        st.subheader("📊 Prediction Results")

        st.write(f"**Risk Score:** {risk_score:.3f}")
        st.metric("Confidence", f"{confidence:.2f}%")

        if "High" in risk:
            st.error(f"⚠️ {risk}")
        elif "Moderate" in risk:
            st.warning(f"⚠️ {risk}")
        else:
            st.success(f"✅ {risk}")

    except Exception as e:
        st.error(f"Error: {e}")
