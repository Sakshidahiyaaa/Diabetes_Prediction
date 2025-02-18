import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))  # Load the saved scaler

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input_data to a NumPy array and reshape
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)

    # Scale the input data using the same scaler
    input_data_scaled = scaler.transform(input_data_as_numpy_array)

    # Make prediction
    prediction = loaded_model.predict(input_data_scaled)

    # Return result
    if prediction[0] == 0:
        return "The person is NOT diabetic"
    else:
        return "The person IS diabetic"

# Streamlit Web App
def main():
    # Title
    st.title("Diabetes Prediction Web App")

    # Input fields
    Pregnancies = st.text_input("Number of Pregnancies", "0")
    Glucose = st.text_input("Glucose Level", "0")
    BloodPressure = st.text_input("Blood Pressure value", "0")
    SkinThickness = st.text_input("Skin Thickness value", "0")
    Insulin = st.text_input("Insulin Level", "0")
    BMI = st.text_input("BMI value", "0")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value", "0")
    Age = st.text_input("Age of the Person", "0")

    # Variable to store the prediction result
    diagnosis = ""

    # Button for Prediction
    if st.button("Diabetes Test Result"):
        # Convert inputs to float before passing to the function
        input_data = [
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]

        diagnosis = diabetes_prediction(input_data)

    # Display the result
    st.success(diagnosis)

# Run the app
if __name__ == "__main__":
    main()
