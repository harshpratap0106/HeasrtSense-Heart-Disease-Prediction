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
# Load Models
# -------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "heart_best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model_columns (2).pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "best_threshold.pkl"))

# ✅ Load ANFIS models
anfis_models = [
    joblib.load(os.path.join(BASE_DIR, "anfis_model_0.pkl")),
    joblib.load(os.path.join(BASE_DIR, "anfis_model_1.pkl")),
    joblib.load(os.path.join(BASE_DIR, "anfis_model_2.pkl"))
]

# -------------------------
# Sidebar Input
# -------------------------
st.sidebar.header("🩺 Enter Patient Details")

def user_input():
    data = {}

    data["age"] = st.sidebar.slider("Age", 20, 100, 40)

    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    data["sex"] = 1 if sex == "Male" else 0

    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    data["cp"] = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

    data["trestbps"] = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    data["chol"] = st.sidebar.slider("Cholesterol", 100, 400, 200)

    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    data["fbs"] = 1 if fbs == "Yes" else 0

    restecg = st.sidebar.selectbox(
        "Resting ECG",
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
    )
    data["restecg"] = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

    data["thalach"] = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    data["exang"] = 1 if exang == "Yes" else 0

    data["oldpeak"] = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    data["slope"] = ["Upsloping", "Flat", "Downsloping"].index(slope)

    data["ca"] = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)

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
        # Prepare input
        X = input_df[columns]
        X_scaled = scaler.transform(X)

        # -------------------------
        # 1. Main Model
        # -------------------------
        prob = model.predict_proba(X_scaled)[0][1]

        # -------------------------
        # 2. ANFIS (Interpretability)
        # -------------------------
        anfis_preds = []

        for m in anfis_models:
            try:
                pred = m.forward(X_scaled)[0]
            except:
                # fallback if model stored differently
                pred = m.predict(X_scaled)[0]
            anfis_preds.append(pred)

        anfis_risk = np.mean(anfis_preds)

        # -------------------------
        # 3. IVCFS (Uncertainty)
        # -------------------------
        lower = max(0, anfis_risk - 0.1)
        upper = min(1, anfis_risk + 0.1)
        confidence = (1 - (upper - lower)) * 100

        # -------------------------
        # 4. Risk Classification
        # -------------------------
        if anfis_risk < threshold - 0.1:
            risk = "🟢 Low Risk"
        elif anfis_risk > threshold + 0.1:
            risk = "🔴 High Risk"
        else:
            risk = "🟡 Moderate Risk"

        # -------------------------
        # Output
        # -------------------------
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        st.write(f"**GB Model Risk:** {prob:.3f}")
        st.write(f"**ANFIS Risk:** {anfis_risk:.3f}")
        st.write(f"**Risk Interval:** [{lower:.3f}, {upper:.3f}]")
        st.metric("Confidence", f"{confidence:.2f}%")

        if "High" in risk:
            st.error(risk)
        elif "Moderate" in risk:
            st.warning(risk)
        else:
            st.success(risk)

    except Exception as e:
        st.error(f"Error: {e}")
