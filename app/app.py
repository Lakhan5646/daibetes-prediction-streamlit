
import streamlit as st
import numpy as np
import joblib
import os
st.write(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_daibetes_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter patient details to predict diabetes risk")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi,
                             diabetes_pedigree, age]])

    probability = model.predict_proba(input_data)[0][1]

    if probability < 0.4:
        st.success(f"Low Risk of Diabetes (Score: {probability:.2f})")
    elif probability < 0.7:
        st.warning(f"Medium Risk of Diabetes (Score: {probability:.2f})")
    else:

        st.error(f"High Risk of Diabetes (Score: {probability:.2f})")
