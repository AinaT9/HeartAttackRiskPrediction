import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import shap
from streamlit_shap import st_shap

# Nuevo modelo
modelo_cargado = joblib.load('Modelo/modelo_rf_doctors.joblib')

feature_ranges = [
    (18, 90),      # Age
    (0, 1),        # Gender
    (0, 1),        # Diabetes
    (70, 250),     # Blood sugar
    (100, 400),    # Cholesterol
    (90, 200),     # Systolic BP
    (60, 130),     # Diastolic BP
    (0, 1),        # Smoking
    (0, 1),        # Alcohol Consumption
    (0, 20),       # Exercise hours/week
    (0, 1),        # Medication Use
    (0, 1),        # Previous Heart Problems
]
columns = [
    'Age', 'Gender', 'Diabetes', 'Blood sugar', 'Cholesterol',
    'Systolic blood pressure', 'Diastolic blood pressure', 'Smoking',
    'Alcohol Consumption', 'Exercise Hours Per Week',
    'Medication Use', 'Previous Heart Problems'
]

def show_dashboard():
    st.header("Doctor View: Predict Heart Attack Risk")
    st.subheader("Enter clinical and lifestyle data:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(
            label="Age",
            min_value=18,
            max_value=90
        )
    with col2:
        gender = st.radio("Gender", ["Male", "Female"])
        gender = 1 if gender == "Male" else 0
    with col3:
        diabetes = st.radio("Diabetes", ["Yes", "No"])

    col4, col5, col6 = st.columns(3)
    with col4:
        blood_sugar = st.number_input(
            label="Blood Sugar (mg/dL)",
            min_value=70,
            max_value=250
        )
    with col5:
        cholesterol = st.number_input(
            label="Cholesterol (mg/dL)",
            min_value=100,
            max_value=400
        )
    with col6:
        sbp = st.number_input(
            label="Systolic BP (mmHg)",
            min_value=90,
            max_value=200
        )

    col7, col8, col9 = st.columns(3)
    with col7:
        dbp = st.number_input(
            label="Diastolic BP (mmHg)",
            min_value=60,
            max_value=130
        )
    with col8:
        smoking = st.radio("Smoking", ["Yes", "No"])
    with col9:
        alcohol = st.radio("Alcohol Consumption", ["Yes", "No"])

    col10, col11, col12 = st.columns(3)
    with col10:
        exercise = st.number_input(
            label="Exercise (hrs/week)",
            min_value=0,
            max_value=20
        )
    with col11:
        medication = st.radio("Medication Use", ["Yes", "No"])
    with col12:
        previous_heart = st.radio("Previous Heart Problems", ["Yes", "No"])

    _, col_button, _ = st.columns(3)
    with col_button:
        if st.button("CALCULATE!"):
            pred = [
                age, gender, diabetes, blood_sugar, cholesterol, sbp, dbp,
                smoking, alcohol, exercise, medication, previous_heart
            ]
            st.subheader("Estimated Risk:")
            value = get_prediction(pred)

            if value < 30:
                color = "green"
                image_path = "images/SemaforoVerde.png"
            elif 30 <= value < 70:
                color = "yellow"
                image_path = "images/SemaforoAmarillo.png"
            else:
                color = "red"
                image_path = "images/SemaforoRojo.png"

            col1, col2 = st.columns([1, 1])
            with col1:
                image = Image.open(image_path)
                st.image(image, width=250)
            with col2:
                st.markdown(
                    f"""<div style='
                        background-color: {color};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 24px;
                        font-weight: bold;
                        color: black;
                        width: 200px;
                        margin: auto;
                    '>
                        {value} %
                    </div>
                    """, unsafe_allow_html=True)

            # SHAP Explanation
            pred_processed = preprocess_input(pred)
            explainer = shap.Explainer(modelo_cargado)
            shap_values = explainer(pred_processed)
            shap_value = shap.Explanation(
                values=shap_values.values[:, 1],
                base_values=shap_values.base_values[0][1] * 100,
                data=shap_values.data,
                feature_names=columns
            )
            st.write("### Specific Explanation")
            st_shap(shap.plots.force(shap_value, matplotlib=True))
            plt.show()

def preprocess_input(pred):
    new_pred = []
    for i, p in enumerate(pred):
        if p == 'No':
            p = 0
        elif p == 'Yes':
            p = 1
        else:
            p = normalize(p, feature_ranges[i])
        new_pred.append(p)
    return np.array(new_pred).reshape(1, -1)

def normalize(p, range_values):
    value = float(p)
    min_val, max_val = range_values
    norm = (value - min_val) / (max_val - min_val)
    return max(0, min(1, norm))

def get_prediction(pred):
    pred_processed = preprocess_input(pred)
    probab = modelo_cargado.predict_proba(pred_processed)
    return int(probab[0][1] * 100)
