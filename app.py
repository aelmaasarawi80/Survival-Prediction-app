import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and preprocessor
model = joblib.load("rsf_model_compressed.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Tooth Survival Prediction App")

# Input form
age = st.number_input("Age", min_value=10, max_value=100, step=1)
vitality = st.selectbox("Vitality", ["Vital", "Nonvital"])
gender = st.selectbox("Gender", ["Male", "Female"])
protocol = st.selectbox("Protocol", ["Protocol 1", "Protocol 2", "Protocol 3"])
toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
provider = st.selectbox("Provider", ["Specialist", "General Practitioner"])
visits = st.radio("Visits", ["Single", "Multiple"])
prct = st.number_input("PRCT", format="%.3f")
pdttts = st.number_input("PDttts", format="%.3f")

# Fix capitalization/formatting issues permanently
normalize_input = {
    "vitality": {"Vital": "Vital", "Nonvital": "Nonvital"},
    "gender": {"Male": "Male", "Female": "Female"},
    "Protocol": {
        "Protocol 1": "Protocol 1",
        "Protocol 2": "protocol 2",  # lowercase p
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

if st.button("Predict Survival"):
    try:
        # Apply consistent formatting
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

        st.subheader("🧾 Input Summary")
        st.json(patient_data)

        # Convert to DataFrame
        X_new = pd.DataFrame([patient_data])
        X_encoded = preprocessor.transform(X_new)

        # Predict survival
        surv = model.predict_survival_function(X_encoded)[0]
        time_points = [1, 3, 5, 10, 15, 20]

        st.subheader("📊 Predicted Survival Probabilities")
        for t in time_points:
            if t <= max(surv.x):
                st.write(f"{t}-year: **{surv(t):.2%}**")

        # Plot survival curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(surv.x, surv(surv.x), where="post", label="Survival Curve")
        for t in time_points:
            if t <= max(surv.x):
                ax.plot(t, surv(t), "ro")
                ax.text(t, surv(t), f"{surv(t):.2%}", ha="center", va="bottom")
        ax.set_xlabel("Years")
        ax.set_ylabel("Survival Probability")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)
