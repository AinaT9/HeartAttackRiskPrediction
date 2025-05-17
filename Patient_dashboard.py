import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
import joblib
import numpy
import shap
from streamlit_shap import st_shap
import dalex as dx
from sklearn.impute import KNNImputer

feature_ranges= [
    (0, 1),     # Diabetes
    (0, 20),    # Exercise Hours Per Week
    (0, 1),     # Medication Use
    (18, 90),   # Age
    (0, 1),     # Previous Heart Problems
    (0, 1),     # Gender
    (0, 1),     # Smoking
    (0, 1),     # Alcohol Consumption
    (0, 12),    # Sleep Hours Per Day
    (0, 1)      # Diet
]

columns =  [
    'Diabetes',
    'Exercise Hours Per Week',
    'Medication Use',
    'Age',
    'Previous Heart Problems',
    'Gender', 
    'Smoking',
    'Alcohol Consumption',
    'Sleep Hours Per Day',
    'Diet'
]

modelo_cargado = joblib.load('Modelo/modelo_rf_patients.joblib')

def show_dashboard():
    st.header("What is your risk of heart attack?")
    st.subheader("Please complete the following fields:")
    pred = []
    col1, col2, col3 = st.columns(3)
    with col1: 
        sex = st.segmented_control("Sex", ["M", "F"])
        sex = 0 if sex == "M" else 1
        pred.append(sex)
    with col2:
        age = st.number_input(
            label="Introduce your age",
            min_value=18,
            max_value=90
        )
        pred.append(age)
    with col3:
        diabetes = st.selectbox("Do you have diabetes?",("No", "Yes"))
        pred.append(diabetes)

       

    col4, col5, col6 = st.columns(3)
    with col4:
        alcohol = st.selectbox("Do you drink alcohol?",("No", "Yes"))
        pred.append(alcohol)

    with col5:
        medication = st.selectbox("Do you take any medication?",("No", "Yes"))
        pred.append(medication)

    with col6:
        smoking = st.selectbox("Do you smoke?",("No", "Yes"))
        pred.append(smoking)

    col7, col8, col9 = st.columns(3)
    with col7:
        exercise = st.number_input(
            label="How many hours a week do you exercise?",
            min_value=0,
            max_value=20
        )
        pred.append(exercise)

    with col8:
        sleep = st.number_input(
            label="How many hours do you sleep per day?",
            min_value=0,
            max_value=12
        )
        pred.append(sleep)
    with col9:
        diet = st.selectbox("Do you do diet?",("No", "Yes"))
        pred.append(diet)
    _, col11, col12 = st.columns(3)
    with col11: 
        val = st.radio('Previous Heart Problems', ["Yes", "No"])
        val = 1 if val == "Yes" else 0
        pred.append(val)
    with col12: 
        b = st.button("CALCULATE!")

    if b:
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
        # Explain the prediction
        X = explain_dashboard()
        explainer = dx.Explainer(modelo_cargado, X)
        local_exp = explainer.predict_parts(pred)

        # Explanation Breakdown Plot
        st.header(f"Breakdown Explanation for your data")
        fig = local_exp.plot(show=False)  
        st.plotly_chart(fig)

        

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



def explain_dashboard():
    df = pd.read_csv("Modelo/data/heart-attack-risk-prediction-dataset.csv")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns) 
    X =  df[columns]
    X= X.sample(n=200, random_state=42)
    return X