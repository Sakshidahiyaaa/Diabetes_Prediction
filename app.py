import os
import pickle
import numpy as np
import streamlit as st

# Check if model files exist
if os.path.exists("trained_model.sav") and os.path.exists("scaler.sav"):
    loaded_model = pickle.load(open("trained_model.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))
else:
    st.error("Error: Model files not found! Please upload trained_model.sav and scaler.sav")

# Streamlit UI
st.title("Diabetes Prediction Web App")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure value", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness value", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI value", min_value=0.0, max_value=100.0, value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function value", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("Age of the Person", min_value=0, max_value=120, value=0)

# Prediction button
if st.button("Diabetes Test Result"):
    if os.path.exists("trained_model.sav") and os.path.exists("scaler.sav"):
        # Preprocess the input
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_data_scaled = scaler.transform(input_data)

        # Get prediction
        prediction = loaded_model.predict(input_data_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"Prediction: {result}")
    else:
        st.error("Model files not found! Cannot make prediction.")
