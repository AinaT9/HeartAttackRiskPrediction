import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
import joblib
import numpy
import shap
from streamlit_shap import st_shap

feature_ranges = [
    (0, 1),     # Gender
    (18, 90),   # Age
    (0, 1),     # Diabetes
    (0, 1),     # Alcohol Consumption
    (0, 1),     # Medication
    (0, 1),     # Smoking
    (0, 1),     # Diet
    (0, 20),    # Exercise hours/week
    (0, 12),    # Sleep hours/day
]
columns = ['Sex', 'Age', 'Diabetes', 'Alcohol', 'Medication', 'Smoking', 'Diet', 'Exercise hours/week', 'Sleep hours/day']
modelo_cargado = joblib.load('Modelo/modelo_rf_patients.joblib')

def show_dashboard():
    st.header("What is your risk of heart attack?")
    st.subheader("Please complete the following fields:")

    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.segmented_control("Sex", ["M", "F"])
        sex = 0 if sex == "M" else 1
    with col2:
        age = st.slider("Select Age", 18, 90, 55)
    with col3:
        diabetes = st.selectbox("Do you have diabetes?",("No", "Yes"))
       

    col4, col5, col6 = st.columns(3)
    with col4:
        alcohol = st.selectbox("Do you drink alcohol?",("No", "Yes"))
    with col5:
        medication = st.selectbox("Do you take any medication?",("No", "Yes"))
    with col6:
        smoking = st.selectbox("Do you smoke?",("No", "Yes"))

    col7, col8, col9 = st.columns(3)
    with col7:
        exercise =  st.slider("How many hours a week do you exercise?",0, 20, 5)
    with col8:
        sleep = st.slider("How many hours do you sleep per day?",0, 12, 7)
    with col9:
        diet = st.selectbox("Do you do diet?",("No", "Yes"))

    _, col11, col12 = st.columns(3)
    with col11: 
        heart_problems = st.selectbox("Did you have any previous heart problems?",("No", "Yes"))
    with col12: 
        b = st.button("CALCULATE!")

    if b:
        pred = [diabetes,exercise,medication,age,heart_problems,sex, smoking, alcohol, sleep, diet]
        st.subheader("Your risk is")

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
            rotated_image = image.rotate(90, expand=True)
            st.image(rotated_image, width=250)
        with col2:
            st.markdown(
                f"""
                <div style='
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
                """,
                unsafe_allow_html=True
            )
        
        explainer = shap.Explainer(modelo_cargado)
        pred = preprocess_input(pred)
        shap_values = explainer(pred)
        shap_value = shap.Explanation(
            values=shap_values.values[:, 1],         
            base_values=shap_values.base_values[0][1] *100,   
            data=shap_values.data,                    
            feature_names=columns              
        )
        st.write("### Specific Explanation")
        st_shap(shap.plots.force(shap_value, matplotlib=True)) 
        plt.show()

        

def preprocess_input(pred):
    new_pred = []
    for i,p in enumerate(pred): 
        if p == 'No':
            p = 0 
        elif p == 'Yes':
            p = 1
        else: 
            p = normalize(p, feature_ranges.__getitem__(i))
        new_pred.append(p)
    return numpy.array(new_pred)

def normalize(p, range_values):
    value = float(p)
    min_val, max_val = range_values
    norm = (value - min_val) / (max_val - min_val)
    return max(0, min(1, norm))


def get_prediction(pred):
    st.text(pred)
    pred = preprocess_input(pred)
    pred= pred.reshape(1, -1)
    probab = modelo_cargado.predict_proba(pred)
    return int(probab[0][1] * 100)


