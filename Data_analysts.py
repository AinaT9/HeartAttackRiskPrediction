import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import shap
from sklearn.impute import KNNImputer
from streamlit_shap import st_shap

modelo_cargado = joblib.load("Modelo/modelo_rf_datascientists.joblib")

columns = [
    'Age', 'Gender', 'Diabetes', 'Blood sugar', 'Cholesterol',
    'Triglycerides', 'BMI', 'Systolic blood pressure',
    'Diastolic blood pressure', 'Smoking', 'Alcohol Consumption',
    'Obesity', 'CK-MB', 'Troponin', 'Stress Level'
]

feature_ranges = [
    (18, 90),     # Age
    (0, 1),       # Gender
    (0, 1),       # Diabetes
    (70, 250),    # Blood sugar
    (100, 400),   # Cholesterol
    (50, 500),    # Triglycerides
    (15, 50),     # BMI
    (90, 200),    # SBP
    (60, 130),    # DBP
    (0, 1),       # Smoking
    (0, 1),       # Alcohol
    (0, 1),       # Obesity
    (0, 100),     # CK-MB (ng/mL)
    (0, 100),     # Troponin (ng/mL)
    (0, 10)       # Stress level
]

def show_dashboard():
    st.header("Data Scientist Dashboard: Model Testing Interface")
    st.subheader("Most important features")
    explainer, X = explain_dashboard()
    shap_values = explainer(X)[:,:,1]
    st_shap(shap.plots.beeswarm(shap_values))

    st_shap(shap.plots.scatter(shap_values[:, 'Cholesterol']))

    st.subheader("Enter patient data for ML model evaluation")
    inputs = []
    for i, col in enumerate(columns):
        if col in ['Gender', 'Diabetes', 'Smoking', 'Alcohol Consumption', 'Obesity']:
            val = st.radio(f"{col}", ["Yes", "No"])
            val = 1 if val == "Yes" else 0
        else:
            min_val, max_val = feature_ranges[i]
            val = st.slider(f"{col}", min_val, max_val, int((min_val + max_val) / 2))
        inputs.append(val)

    if st.button("Predict and Explain"):
        st.subheader("Predicted Heart Attack Risk:")
        risk = get_prediction(inputs)
        display_risk_indicator(risk)

        # SHAP
        inputs_arr = np.array(inputs).reshape(1, -1)
        explainer = shap.Explainer(modelo_cargado)
        shap_values = explainer(inputs_arr)
        shap_value = shap.Explanation(
            values=shap_values.values[:, 1],
            base_values=shap_values.base_values[0][1] * 100,
            data=shap_values.data,
            feature_names=columns
        )
        st.write("### SHAP Explanation (Model Interpretation)")
        st_shap(shap.plots.force(shap_value, matplotlib=True))
        plt.show()

def explain_dashboard():
    df = pd.read_csv("Modelo/data/heart-attack-risk-prediction-dataset.csv")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns) 
    X =  df[columns]
    explainer = shap.Explainer(modelo_cargado)
    X= X.sample(n=200, random_state=42)
    return explainer, X

def get_prediction(pred):
    pred = np.array(pred).reshape(1, -1)
    probab = modelo_cargado.predict_proba(pred)
    return int(probab[0][1] * 100)

def display_risk_indicator(value):
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
