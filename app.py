import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load  model/preprocessor
model = joblib.load("rsf_model_compressed.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Tooth Survival Prediction")

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

if st.button("Predict Survival"):
    # Normalize category values
    norm = {
       "Protocol 2": "protocol 2", # lowercase
       "Anterior": "Anterior tooth", # note mismatch with preprocessor
    }
    pd_input = {
       "age": age,
       "vitality": vitality,
       "gender": gender,
       "Protocol": norm.get(protocol, protocol),
       "toothtype": norm.get(toothtype, toothtype),
       "provider": provider,
       "Visits": visits,
       "PRCT": prct,
       "PDttts": pdttts,
    }

    st.write("🔍 Raw input:", pd_input)

    try:
        X_new = pd.DataFrame([pd_input])
        st.write("➡️ DataFrame before transform:", X_new)

        X_enc = preprocessor.transform(X_new)
        st.write("✅ Encoded shape:", X_enc.shape)

        surv = model.predict_survival_function(X_enc)[0]
        st.write("Surv.x:", surv.x[:5], "…", surv.x[-5:])
        st.write("Surv(surv.x):", surv(surv.x)[:5], "…")

        # Choose timepoints
        times = [1,3,5,10,15,20]
        valid = {t: surv(t) for t in times if t <= max(surv.x)}
        st.write("📊 Survival probs:", valid)

        fig, ax = plt.subplots()
        ax.step(surv.x, surv(surv.x), where='post')
        for t,p in valid.items():
            ax.plot(t, p, "ro")
        st.pyplot(fig)

    except Exception as ex:
        st.error("❌ Error during prediction — see details 📋")
        st.exception(ex)
