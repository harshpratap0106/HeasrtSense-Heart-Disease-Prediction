import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="HeartSense", layout="centered")

st.title("HeartSense: Heart Disease Prediction System")

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

# ==========================================
# 4. PREDICTION LOGIC (FIXED)
# ==========================================
if submit_button:

    try:
        # A. Initialize all features to 0
        input_dict = {col: 0 for col in model_columns}

        # B. Numeric features
        input_dict['Age'] = age
        input_dict['RestingBP'] = resting_bp
        input_dict['Cholesterol'] = cholesterol
        input_dict['FastingBS'] = 1 if fbs == "Yes" else 0
        input_dict['MaxHR'] = max_hr
        input_dict['Oldpeak'] = oldpeak

        # C. Categorical Encoding (SAFE MAPPING)

        # Sex
        if sex == "Male" and "Sex_M" in input_dict:
            input_dict["Sex_M"] = 1

        # Chest Pain
        cp_map = {
            "Typical Angina": "TA",
            "Atypical Angina": "ATA",
            "Non-anginal Pain": "NAP",
            "Asymptomatic": "ASY"
        }
        cp_code = cp_map[chest_pain]
        col_name = f"ChestPainType_{cp_code}"
        if col_name in input_dict:
            input_dict[col_name] = 1

        # ECG
        ecg_map = {
            "Normal": "Normal",
            "ST-T Abnormality": "ST",
            "Left Ventricular Hypertrophy": "LVH"
        }
        ecg_code = ecg_map[ecg]
        col_name = f"RestingECG_{ecg_code}"
        if col_name in input_dict:
            input_dict[col_name] = 1

        # Exercise Angina
        if angina == "Yes" and "ExerciseAngina_Y" in input_dict:
            input_dict["ExerciseAngina_Y"] = 1

        # Slope
        slope_map = {
            "Upsloping": "Up",
            "Flat": "Flat",
            "Downsloping": "Down"
        }
        slope_code = slope_map[slope]
        col_name = f"ST_Slope_{slope_code}"
        if col_name in input_dict:
            input_dict[col_name] = 1

        # D. DataFrame
        final_df = pd.DataFrame([input_dict])[model_columns]

        # E. Scaling
        scaled_data = scaler.transform(final_df)

        # F. Prediction
        risk_probability = model.predict_proba(scaled_data)[0][1] * 100

        # ==========================================
        # 5. DISPLAY RESULTS
        # ==========================================
        st.markdown("---")
        st.subheader(f"Results for Patient (Age {age})")

        st.write(f"**Predicted Heart Disease Risk Score:** {risk_probability:.1f}%")
        st.progress(int(risk_probability))

        if risk_probability < 30:
            st.success("Level: **Low Risk** ✅")
        elif risk_probability < 60:
            st.warning("Level: **Moderate Risk** ⚠️")
        else:
            st.error("Level: **High Risk** 🚨")

        with st.expander("See Clinical Insights"):
            st.write(
                "This score represents the probability of heart disease based on clinical markers. "
                "A higher score indicates a pattern commonly observed in patients with heart disease."
            )

    except Exception as e:
        st.error(f"Error in prediction: {e}")

        # Confidence
        confidence = (1 - abs(risk_score - threshold)) * 100

