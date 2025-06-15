import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Helper function to load files from Google Drive
def load_from_drive(url):
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    return BytesIO(response.content)

# Load model and preprocessor from Google Drive 
@st.cache_resource
def load_model():
    # Use YOUR Google Drive links here
    survival_model_url = "https://drive.google.com/file/d/1_mFMn9uqLRy9KDg8M7nS0fwhLRacS2v7/view?usp=drive_link"
    preprocessor_url = "https://drive.google.com/file/d/1iTBD5o8-xPlkiAI-qtXOKT1q7CMmbm6q/view?usp=drive_link"

    model_file = load_from_drive(survival_model_url)
    preprocessor_file = load_from_drive(preprocessor_url)

    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)
    return model, preprocessor

model, preprocessor = load_model()

# Title and description 
st.title("🦷 Tooth Survival Prediction Tool")
st.write("Enter patient details below to predict tooth survival probability over time.")

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

# Make prediction
if st.button("Predict Survival"):
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

    try:
        # Encode and scale features
        X_encoded = preprocessor.transform(patient_data)

        # Predict survival function
        surv_funcs = model.predict_survival_function(X_encoded)
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
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")