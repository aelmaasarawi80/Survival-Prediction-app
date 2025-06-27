import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# Load the model and preprocessor
model = joblib.load("rsf_model_compressed.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Tooth Survival Prediction After Root Canal Treatment")

# --- INPUT FIELDS ---
age = st.number_input("Patient Age", min_value=10, max_value=100, step=1)
vitality = st.selectbox("Vitality", ["Vital", "Nonvital"])
gender = st.selectbox("Gender", ["Male", "Female"])
protocol = st.selectbox("Protocol", ["Protocol 1", "Protocol 2", "Protocol 3"])
toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
provider = st.selectbox("Provider", ["Specialist", "General Practitioner"])
visits = st.radio("Number of Visits", ["Single", "Multiple"])
prct = st.number_input("PRCT (e.g., crown length)", format="%.3f")
pdttts = st.number_input("PDttts (e.g., probing depth)", format="%.3f")

# Normalize categorical inputs to match what model saw
normalize_input = {
    "vitality": {"Vital": "Vital", "Nonvital": "Nonvital"},
    "gender": {"Male": "Male", "Female": "Female"},
    "Protocol": {
        "Protocol 1": "Protocol 1",
        "Protocol 2": "protocol 2",  # Lowercase to match preprocessor
        "Protocol 3": "Protocol 3",
    },
    "toothtype": {
        "Anterior": "Anterior tooth",
        "Premolar": "Premolar",
        "Molar": "Molar",
    },
    "provider": {
        "Specialist": "Specialist",
        "General Practitioner": "General Practitioner",
    },
    "Visits": {"Single": "Single", "Multiple": "Multiple"},
}

# Assemble patient data
patient_data = {
    "age": age,
    "vitality": normalize_input["vitality"][vitality],
    "gender": normalize_input["gender"][gender],
    "Protocol": normalize_input["Protocol"][protocol],
    "toothtype": normalize_input["toothtype"][toothtype],
    "provider": normalize_input["provider"][provider],
    "Visits": normalize_input["Visits"][visits],
    "PRCT": prct,
    "PDttts": pdttts,
}

# Prediction Button
if st.button("Predict Tooth Survival"):

    try:
        # Convert to DataFrame and transform
        X_new = pd.DataFrame([patient_data])
        X_encoded = preprocessor.transform(X_new)

        # Predict survival function
        surv_func = model.predict_survival_function(X_encoded)[0]

        # Evaluate at specific years
        time_points = [1, 3, 5, 10, 15, 20]
        probs = {t: surv_func(t) for t in time_points if t <= max(surv_func.x)}

        st.subheader("Predicted Survival Probabilities:")
        for t, p in probs.items():
            st.write(f"**{t} years**: {p:.2%}")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func.x, surv_func(surv_func.x), where="post", color="blue", label="Survival Curve")
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Predicted Tooth Survival")
        ax.grid(True)

        # Add year markers
        for t in time_points:
            if t <= max(surv_func.x):
                ax.plot(t, surv_func(t), "o", color="red")
                ax.text(t, surv_func(t), f"{surv_func(t):.2%}", ha='center', va='bottom')

        st.pyplot(fig)

    except Exception as e:
        st.error("Prediction failed. Please check the input or model compatibility.")
        st.exception(e)
