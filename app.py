import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
from io import BytesIO

# Load model from GitHub raw link
@st.cache_resource
def load_model(model_url):
    response = requests.get(model_url)
    model_file = BytesIO(response.content)
    model = joblib.load(model_file)
    return model

# Define input form and prediction logic
st.title("🦷 Tooth Survival Prediction Tool")
st.write("Enter patient details below to predict tooth survival probability over time.")

# Paste your GitHub raw link for gbsa_model.pkl
model_url = st.secrets["model_url"] if "model_url" in st.secrets else "https://github.com/aelmaasarawi80/Survival-Prediction-app/blob/main/survival_model.pkl"

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=40)
vitality = st.selectbox("Vitality", ['Vital', 'Nonvital', 'Unknown'])
gender = st.selectbox("Gender", ['Male', 'Female'])
protocol = st.selectbox("Protocol", ['Hand Files', 'Reciproc', 'RaCe-BioRaCe'])
toothtype = st.selectbox("Tooth Type", ['Molar', 'Premolar', 'Anterior tooth'])
provider = st.selectbox("Provider", ['Specialist', 'General Practitioner'])
visits = st.selectbox("Visits", ['Single', 'Multiple'])
prct = st.slider("PRCT (0–3)", min_value=0, max_value=3, value=0)
no_visits = st.number_input("Number of Visits", min_value=1, max_value=10, value=2)
time_before_obturation = st.number_input("Time Before Obturation (days)", min_value=0, max_value=100, value=7)
pdttts = st.number_input("PDttts (Periodontal Depth)", min_value=0, max_value=30, value=0)

if st.button("Predict Survival"):
    try:
        # Load model from GitHub
        model = load_model(model_url)

        # Create DataFrame from input
        patient_data = pd.DataFrame([{
            'age': age,
            'vitality': vitality,
            'gender': gender,
            'Protocol': protocol,
            'toothtype': toothtype,
            'provider': provider,
            'Visits': visits,
            'PRCT': prct,
            'No_Visits': no_visits,
            'time before obturation': time_before_obturation,
            'PDttts': pdttts
        }])

        # Encode features using same preprocessor used during training
        # ❗ This assumes you have a fixed preprocessor or encoded inputs match training data

        # Predict survival function
        surv_funcs = model.predict_survival_function(patient_data)
        surv_func = surv_funcs[0]

        # Time points
        time_points = [1, 3, 5, 10]
        probs = [surv_func(t) for t in time_points]

        # Display results
        st.subheader("📊 Predicted Survival Probabilities")
        for t, p in zip(time_points, probs):
            st.write(f"{t}-year survival probability: **{p:.2%}**")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func.x, surv_func(surv_func.x), where="post")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival Probability")
        ax.grid(True)

        for t in time_points:
            if t <= max(surv_func.x):
                ax.plot(t, surv_func(t), "o", color="red")
                ax.text(t, surv_func(t), f"{surv_func(t):.2%}", ha='center', va='bottom')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
