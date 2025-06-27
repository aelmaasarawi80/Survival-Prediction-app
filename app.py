import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and preprocessor
model = joblib.load("rsf_model_compressed.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Tooth Survival Prediction After Root Canal Treatment")

# User inputs
age = st.slider("Age", 18, 90, 30)
vitality = st.selectbox("Pulp Vitality", ["Vital", "Nonvital"])
gender = st.selectbox("Gender", ["Male", "Female"])
protocol = st.selectbox("Treatment Protocol", ["Protocol 1", "Protocol 2", "Protocol 3"])
toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
provider = st.selectbox("Provider Type", ["Specialist", "General Practitioner"])
visits = st.radio("Number of Visits", ["Single", "Multiple"])
prct = st.slider("PRCT (mm)", 0, 15, 5, step=1)         # Changed to int-only
pdttts = st.slider("PDttts (mm)", 0, 50, 10, step=1)    # Changed to int-only

# Input normalization
normalize_input = {
    "vitality": {"Vital": "Vital", "Nonvital": "Nonvital"},
    "gender": {"Male": "Male", "Female": "Female"},
    "Protocol": {
        "Protocol 1": "Protocol 1",
        "Protocol 2": "protocol 2",  # match training lowercase
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
}

patient_data = {
    "age": age,
    "vitality": normalize_input["vitality"][vitality],
    "gender": normalize_input["gender"][gender],
    "Protocol": normalize_input["Protocol"][protocol],
    "toothtype": normalize_input["toothtype"][toothtype],
    "provider": normalize_input["provider"][provider],
    "Visits": visits,
    "PRCT": prct,
    "PDttts": pdttts,
}

if st.button("Predict Survival"):
    try:
        X_new = pd.DataFrame([patient_data])
        X_encoded = preprocessor.transform(X_new)
        surv_funcs = model.predict_survival_function(X_encoded)
        surv_func = surv_funcs[0]

        st.subheader("Predicted Survival Probabilities:")
        for t in [1, 3, 5, 10, 15, 20]:
            try:
                st.write(f"{t}-year survival: **{surv_func(t):.2%}**")
            except:
                st.write(f"{t}-year survival: value unavailable")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func.x, surv_func(surv_func.x), where="post", color="blue")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Estimated Survival Probability")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_ylim(0, 1.05)  # 🔧 Y-axis starts at 0
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
