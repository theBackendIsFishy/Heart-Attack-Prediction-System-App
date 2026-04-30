import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("Heart_Attack_Prediction.pkl")

st.title("❤️ Heart Attack Prediction App")

st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age")
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
restecg = st.number_input("Rest ECG (0-2)")
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope (0-2)")
ca = st.number_input("Number of Major Vessels (0-3)")
thal = st.number_input("Thal (0-3)")

if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        st.error(f"High Risk of Heart Attack ⚠️ ({round(probability[1]*100,2)}%)")
    else:
        st.success(f"Low Risk of Heart Attack ✅ ({round(probability[0]*100,2)}%)")