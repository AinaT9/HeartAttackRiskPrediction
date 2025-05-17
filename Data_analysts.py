import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import shap
from sklearn.impute import KNNImputer
from streamlit_shap import st_shap
from sklearn.preprocessing import LabelEncoder


modelo_cargado = joblib.load("Modelo/modelo_rf.joblib")

columns= [
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
    "Country",
    "Systolic blood pressure",
    "Diastolic blood pressure"
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
    # for i, col in enumerate(columns):
    #     if col in ['Gender', 'Diabetes', 'Smoking', 'Alcohol Consumption', 'Obesity']:
    #         val = st.radio(f"{col}", ["Yes", "No"])
    #         val = 1 if val == "Yes" else 0
    #     else:
    #         val = st.slider(f"{col}", min_val, max_val, int((min_val + max_val) / 2))
    #     inputs.append(val)

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
    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.drop(columns=['Patient ID', 'Income','Physical Activity Days Per Week','Continent', 'Hemisphere'])
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
    df['Diet'] = df['Diet'].map(diet_map)
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic blood pressure'] = pd.to_numeric(bp_split[0])
    df['Diastolic blood pressure'] = pd.to_numeric(bp_split[1])
    df = df.drop(columns=['Blood Pressure'])
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    
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
