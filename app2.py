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

# Load model and preprocessor 2
@st.cache_resource
def load_model_and_preprocessor2():
    model2 = joblib.load("gbsa_model.pkl")
    preprocessor2 = joblib.load("preprocessor2.pkl")
    return model2, preprocessor2

model2, preprocessor2 = load_model_and_preprocessor2()

# Load model and preprocessor 3
@st.cache_resource
def load_model_and_preprocessor2():
    model3 = joblib.load("gbsa_model2.pkl")
    preprocessor3 = joblib.load("preprocessor3.pkl")
    return model3, preprocessor3

model3, preprocessor3 = load_model_and_preprocessor2()

# Title
st.title("🦷 Tooth Survival Prediction App")

st.markdown("Enter patient information below to predict tooth survival after root canal treatment.")

# Form inputs
age = st.slider("Age of the patient", min_value=10, max_value=90, value=35)
gender = st.selectbox("Gender of the patient", ["Male", "Female"])
vitality = st.selectbox("Preoperative tooth vitality condition", ["Vital", "Nonvital"])
protocol = st.selectbox("Treatment Protocol implemented", ["Protocol 1", "Protocol 2", "Protocol 3"])
toothtype = st.selectbox("Tooth Type", ["Anterior", "Premolar", "Molar"])
provider = st.selectbox("Provider Type", ["Specialist", "General Practitioner"])
visits = st.selectbox("Number of Visits", ["Single", "Multiple"])
prct = st.slider("No. of previous RCT", min_value=0, max_value=15, value=0)
pdttts = st.slider("No. of previous supportive periodontal treatments", min_value=0, max_value=50, value=0)

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
if st.button("Predict Survival until extraction"):
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
        st.subheader("📈 Predicted Survival Probabilities until extraction")
        for t in time_points:
            if t <= max(surv_func.x):
                st.write(f"**{t}-year**: {surv_func(t):.2%}")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func.x, surv_func(surv_func.x), where="post", color="blue")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Estimated Survival Probability")
        ax.set_ylim(0, 1.05)  # 🔧 Y-axis starts at 0
        ax.grid(True)
        for t in time_points:
            if t <= max(surv_func.x):
                ax.plot(t, surv_func(t), "ro")
                ax.text(t, surv_func(t), f"{surv_func(t):.2%}", ha='center', va='bottom')
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.exception(e)



# Predict button
if st.button("Predict Survival until NS retreatment"):
    try:
        # Build cleaned input dictionary
        patient_data2 = {
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
        X_new2 = pd.DataFrame([patient_data2])
        X_encoded2 = preprocessor2.transform(X_new2)

        # Predict survival
        surv_func2 = model2.predict_survival_function(X_encoded2)[0]

        # Time points
        time_points = [1, 3, 5, 10, 15, 20]

        # Print predictions
        st.subheader("📈 Predicted Survival Probabilities until non-surgical retreatment")
        for t in time_points:
            if t <= max(surv_func2.x):
                st.write(f"**{t}-year**: {surv_func2(t):.2%}")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func2.x, surv_func2(surv_func2.x), where="post", color="blue")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Estimated Survival Probability")
        ax.set_ylim(0, 1.05)  # 🔧 Y-axis starts at 0
        ax.grid(True)
        for t in time_points:
            if t <= max(surv_func2.x):
                ax.plot(t, surv_func2(t), "ro")
                ax.text(t, surv_func2(t), f"{surv_func2(t):.2%}", ha='center', va='bottom')
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.exception(e)


# Predict button
if st.button("Predict Survival until S retreatment"):
    try:
        # Build cleaned input dictionary
        patient_data3 = {
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
        X_new3 = pd.DataFrame([patient_data3])
        X_encoded3 = preprocessor3.transform(X_new3)

        # Predict survival
        surv_func3 = model3.predict_survival_function(X_encoded3)[0]

        # Time points
        time_points = [1, 3, 5, 10, 15, 20]

        # Print predictions
        st.subheader("📈 Predicted Survival Probabilities until surgical retreatment")
        for t in time_points:
            if t <= max(surv_func3.x):
                st.write(f"**{t}-year**: {surv_func3(t):.2%}")

        # Plot survival curve
        fig, ax = plt.subplots()
        ax.step(surv_func3.x, surv_func3(surv_func3.x), where="post", color="blue")
        ax.set_title("Predicted Tooth Survival Curve")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Estimated Survival Probability")
        ax.set_ylim(0, 1.05)  # 🔧 Y-axis starts at 0
        ax.grid(True)
        for t in time_points:
            if t <= max(surv_func3.x):
                ax.plot(t, surv_func3(t), "ro")
                ax.text(t, surv_func3(t), f"{surv_func3(t):.2%}", ha='center', va='bottom')
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.exception(e)
