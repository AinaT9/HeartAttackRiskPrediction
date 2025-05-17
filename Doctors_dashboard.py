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

columns = [
    '', '', '', 'Blood sugar', '',
    'Systolic blood pressure', 'Diastolic blood pressure', '',
    '', '',
    '', ''
]
column_names = [
    "Age",
    "Gender",
    "Cholesterol",
    "Heart Rate",
    "Diabetes",
    "Family History",
    "Smoking",
    "Obesity",
    "Alcohol Consumption",
    "Exercise Hours Per Week",
    "Diet",
    "Previous Heart Problems",
    "Medication Use",
    "Stress Level",
    "Sedentary Hours Per Day",
    "BMI",
    "Triglycerides",
    "Sleep Hours Per Day",
    "Systolic blood pressure",
    "Diastolic blood pressure"
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
        cholesterol = st.number_input(
            label="Cholesterol (mg/dL)",
            min_value=100,
            max_value=400
        )
        
    col4, col5, col6 = st.columns(3)
    with col4:
        heart_rate = st.number_input(
            label="Heart Rate",
            min_value=40,
            max_value=120
        )
    with col5:
        diabetes = st.radio("Diabetes", ["Yes", "No"])
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
        previous_heart = 1 if previous_heart == "Yes" else 0
    
    col13, col14, col15, col16 = st.columns(4)
    with col13:
        triglycerides = st.number_input(
            label="Triglycerides",
            min_value=20,
            max_value=400
        )
    with col14:
        family_history = st.selectbox("Do you have any family history of heart attacks?",("No", "Yes"))
    with col15:
        obesity = st.radio('Have you obesity?', ["Yes", "No"])
        obesity = 1 if obesity == "Yes" else 0
    with col16:
        diet = st.selectbox("How is your diet?",("Unhealthy", "Normal", "Healthy"))    
        diet_map = {"Unhealthy": 0, "Normal": 1, "Healthy": 2}
        diet = diet_map.get(diet, -1) 
    
    col17, col18, col19, col20 = st.columns(4)
    with col17:
        sleep = st.number_input(
                    label="Sleep h/day",
                    min_value=0,
                    max_value=12
                )        
    with col18:
        stress = st.number_input(
            label="From 1 to 10, which is your stress level?",
            min_value=1,
            max_value=10
        )
    with col19:
        sedentary = st.number_input(
            label="How many hours do you sit?",
            min_value=0,
            max_value=12
        )
    with col20:
        bmi = st.number_input(
            label="BMI",
            min_value=15,
            max_value=50
        )


    _, col_button, _ = st.columns(3)
        
    with col_button:
        if st.button("CALCULATE!"):
            pred = [
                age, gender, cholesterol, heart_rate, family_history, smoking, obesity, diabetes,alcohol,exercise, diet,previous_heart,medication, 
                stress, sedentary, bmi, triglycerides,sleep, sbp, dbp
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
        new_pred.append(p)
    return np.array(new_pred).reshape(1, -1)


def get_prediction(pred):
    pred_processed = preprocess_input(pred)
    probab = modelo_cargado.predict_proba(pred_processed)
    return int(probab[0][1] * 100)
