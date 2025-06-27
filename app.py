import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("rsf_model_compressed.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# Title
st.title("🦷 Tooth Survival Prediction App")

st.markdown("Enter patient information below to predict tooth survival after root canal treatment.")

# Form inputs
age = st.slider("Age", min_value=10, max_value=90, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])
vitality = st.selectbox("Vitality", ["Vital", "Nonvital"])
protocol = st.selectbox("Treatment Protocol", ["Protocol 1", "Protocol 2", "Protocol 3"])
toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
provider = st.selectbox("Provider Type", ["Specialist", "General Practitioner"])
visits = st.selectbox("Number of Visits", ["Single", "Multiple"])
prct = st.number_input("PRCT value", value=0.0)
pdttts = st.number_input("PDttts value", value=0.0)

# Normalization dictionary
normalize_input = {
    "vitality": {"Vital": "Vital", "Nonvital": "Nonvital"},
    "gender": {"Male": "Male", "Female": "Female"},
    "Protocol": {
        "Protocol 1": "Protocol 1",
        "Protocol 2": "protocol 2",  # matches training data casing
        "Protocol 3": "Protocol 3",
    },
    "toothtype": {
        "Anterior": "Anterior tooth",  # matches encoded column
        "Premolar": "Premolar",
        "Molar": "Molar",
    },
    "provider": {
        "Specialist": "Specialist",
        "General Practitioner": "General Practitioner",
    },
}

# Predict button
if st.button("Predict Survival"):
    try:
        # Build cleaned input dictionary
        patient_data = {
            "age": age,
            "gender": normalize_input["gender"][gender],
            "vitality": normalize_input["vitality"][vitality],
            "Protocol": normalize_input["Protocol"][protocol],
            "toothtype": normalize_input["toothtype"][toothtype],
            "provider": normalize_input["provider"][provider],
            "Visits": visits,
            "PRCT": prct,
            "PDttts": pdttts,
        }

        # Convert to DataFrame and transform
        X_new = pd.DataFrame([patient_data])
        X_encoded = preprocessor.transform(X_new)

        # Predict survival
        surv_func = model.predict_survival_function(X_encoded)[0]

        # Time points
        time_points = [1, 3, 5, 10, 15, 20]

        # Print predictions
        st.subheader("📈 Predicted Survival Probabilities")
        for t in time_points:
            if t <= max(surv_func.x):
                st.write(f"**{t}-year**: {surv_func(t):.2%}")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func.x, surv_func(surv_func.x), where="post", color="blue")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Estimated Survival Probability")
        ax.grid(True)
        for t in time_points:
            if t <= max(surv_func.x):
                ax.plot(t, surv_func(t), "ro")
                ax.text(t, surv_func(t), f"{surv_func(t):.2%}", ha='center', va='bottom')
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.exception(e)
