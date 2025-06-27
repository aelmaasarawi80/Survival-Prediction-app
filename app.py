import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load("rsf_model_compressed.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# App title
st.title("Tooth Survival Predictor After Root Canal Treatment")

st.write("""
Enter the patient's baseline information below to estimate the probability of tooth survival at different time points.
""")

# Input form
with st.form("patient_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=35)
    vitality = st.selectbox("Vitality", ["Vital", "Nonvital"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    protocol = st.selectbox("Treatment Protocol", ["Protocol 1", "Protocol 2", "Protocol 3"])
    toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
    provider = st.selectbox("Provider Type", ["Specialist", "General Practitioner"])
    visits = st.number_input("Number of Visits", min_value=1, value=1)
    prct = st.number_input("PRCT", min_value=0.0, step=0.1)
    pdttts = st.number_input("PDttts", min_value=0.0, step=0.1)

    # Fix capitalization/whitespace issues
normalize_input = {
    "vitality": {"Vital": "Vital", "Nonvital": "Nonvital"},
    "gender": {"Male": "Male", "Female": "Female"},
    "Protocol": {
        "Protocol 1": "Protocol 1",
        "Protocol 2": "protocol 2",  # Note lowercase p to match training
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

# Build cleaned patient input
patient_data = {
    "age": age,
    "vitality": normalize_input["vitality"][vitality],
    "gender": normalize_input["gender"][gender],
    "Protocol": normalize_input["Protocol"][protocol],
    "toothtype": normalize_input["toothtype"][toothtype],
    "provider": normalize_input["provider"][provider],
    "Visits": "Single" if visits == 1 else "Multiple",
    "PRCT": prct,
    "PDttts": pdttts,
}

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    patient_data = {
        "age": age,
        "vitality": vitality,
        "gender": gender,
        "Protocol": protocol,
        "toothtype": toothtype,
        "provider": provider,
        "Visits": visits,
        "PRCT": prct,
        "PDttts": pdttts
    }

    X_new = pd.DataFrame([patient_data])
    X_encoded = preprocessor.transform(X_new)
    surv_func = model.predict_survival_function(X_encoded)[0]

    time_points = [1, 3, 5, 10, 15, 20]
    survival_probs = {t: surv_func(t) for t in time_points}

    st.subheader("Predicted Survival Probabilities")
    for t in time_points:
        st.write(f"**{t}-year:** {survival_probs[t]*100:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(surv_func.x, surv_func(surv_func.x), where="post", label="Survival Curve")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Predicted Tooth Survival Curve")
    ax.grid(True)

    for t in time_points:
        if t <= max(surv_func.x):
            ax.plot(t, surv_func(t), "ro")
            ax.text(t, surv_func(t), f"{surv_func(t):.2%}", ha='center', va='bottom')

    st.pyplot(fig)
