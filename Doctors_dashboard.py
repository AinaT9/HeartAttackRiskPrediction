import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import dalex as dx
from PIL import Image
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.graph_objects as go

# Cargar modelo entrenado
modelo_cargado = joblib.load('Modelo/modelo_rf_doctors.joblib')

selected_variables = [
    'Age', 'Gender', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
    'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
    'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
    'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'Sleep Hours Per Day',
    'Systolic blood pressure', 'Diastolic blood pressure'
]

# ------------------------- DASHBOARD PRINCIPAL ------------------------- #
def show_dashboard():
    st.title("🩺 Doctor Dashboard: Heart Attack Risk Assessment")
    st.markdown("Please complete the following clinical fields:")

    st.markdown("### 🔹 Patient Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = 1 if st.radio("Gender", ["Male", "Female"]) == "Male" else 0
    with col2:
        age = st.number_input("Age", 18, 90)
    with col3:
        bmi = st.number_input("BMI", 15, 50)

    st.markdown("### 🔹 Vitals and Lab Results")
    col4, col5, col6 = st.columns(3)
    with col4:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400)
    with col5:
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 120)
    with col6:
        systolic = st.number_input("Systolic BP", 90, 200)

    col7, col8, col9 = st.columns(3)
    with col7:
        diastolic = st.number_input("Diastolic BP", 60, 130)
    with col8:
        triglycerides = st.number_input("Triglycerides", 20, 400)
    with col9:
        sleep = st.number_input("Sleep Hours Per Day", 0, 12)

    st.markdown("### 🔹 Lifestyle & Habits")
    col10, col11, col12 = st.columns(3)
    with col10:
        smoking = 1 if st.radio("Smoking", ["Yes", "No"]) == "Yes" else 0
    with col11:
        alcohol = 1 if st.radio("Alcohol Consumption", ["Yes", "No"]) == "Yes" else 0
    with col12:
        exercise = st.number_input("Exercise Hours/Week", 0, 20)

    col13, col14, col15 = st.columns(3)
    with col13:
        sedentary = st.number_input("Sedentary Hours/Day", 0, 12)
    with col14:
        diet = {"Unhealthy": 0, "Normal": 1, "Healthy": 2}[st.selectbox("Diet", ["Unhealthy", "Normal", "Healthy"])]
    with col15:
        stress = st.number_input("Stress Level (1–10)", 1, 10)

    st.markdown("### 🔹 Medical History")
    col16, col17, col18 = st.columns(3)
    with col16:
        diabetes = 1 if st.radio("Diabetes", ["Yes", "No"]) == "Yes" else 0
    with col17:
        family_history = 1 if st.radio("Family History", ["Yes", "No"]) == "Yes" else 0
    with col18:
        medication = 1 if st.radio("Medication Use", ["Yes", "No"]) == "Yes" else 0

    col19, col20, col21 = st.columns(3)
    with col19:
        previous_heart = 1 if st.radio("Previous Heart Problems", ["Yes", "No"]) == "Yes" else 0
    with col20:
        obesity = 1 if st.radio("Obesity", ["Yes", "No"]) == "Yes" else 0


    # --- Botón centrado
    st.markdown("---")
    _, col_button, _ = st.columns([2, 1, 2])
    with col_button:
        if st.button("🚨 CALCULATE!", use_container_width=True):
            pred = [age, gender, cholesterol, heart_rate, diabetes, family_history,
                    smoking, obesity, alcohol, exercise, diet, previous_heart,
                    medication, stress, sedentary, bmi, triglycerides, sleep, systolic, diastolic]

            risk_percent = int(modelo_cargado.predict_proba(np.array(pred).reshape(1, -1))[0][1] * 100)
            risk_class = int(modelo_cargado.predict(np.array(pred).reshape(1, -1))[0])
            emoji = "🚨" if risk_class == 1 else "🩺"
            status = "HIGH RISK" if risk_class == 1 else "LOW RISK"
            st.markdown(f"### You have {emoji} **{status}** of heart attack.")

            image_path = "images/SemaforoVerde.png" if risk_percent < 30 else \
                         "images/SemaforoAmarillo.png" if risk_percent < 70 else \
                         "images/SemaforoRojo.png"
            color = "green" if risk_percent < 30 else "yellow" if risk_percent < 70 else "red"

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(Image.open(image_path).rotate(90, expand=True), width=250)
            with col2:
                st.markdown(f"""<div style='background-color: {color}; padding: 20px; border-radius: 10px;
                                text-align: center; font-size: 24px; font-weight: bold; color: black;
                                width: 200px; margin: auto;'>{risk_percent} %</div>""",
                            unsafe_allow_html=True)

            # --- DALEX Breakdown
            st.header("🔍 Breakdown Explanation")
            X = explain_dashboard()
            explainer = dx.Explainer(modelo_cargado, X)
            local_exp = explainer.predict_parts(np.array(pred))
            st.plotly_chart(local_exp.plot(show=False))

    st.markdown("---")
    show_visualizations()


# ------------------------- VISUALIZACIONES ------------------------- #
def show_visualizations():
    st.header("📊 Clinical Data Insights")

    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})
    bp = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic blood pressure'] = pd.to_numeric(bp[0])
    df['Diastolic blood pressure'] = pd.to_numeric(bp[1])
    df.drop(columns=["Patient ID", "Country", "Continent", "Hemisphere", "Blood Pressure"], inplace=True)
    df.dropna(inplace=True)
    df["Heart Attack Risk (Text)"] = df["Heart Attack Risk"].map({0: "Low", 1: "High"})

    # Correlation
    corr = df.corr(numeric_only=True).round(2)

    # Age vs Risk
    st.plotly_chart(px.scatter(df, x="Age", y="Heart Attack Risk",
                               trendline="lowess", color="Heart Attack Risk (Text)",
                               title="📈 Age vs Probability of Heart Attack"))

    # Correlation matrix
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1, colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(title="📊 Correlation Matrix (Full Dataset)", width=900, height=750)
    st.plotly_chart(fig_corr)

    # Stress vs Exercise heatmap
    fig_stress = px.density_heatmap(df, x="Stress Level", y="Exercise Hours Per Week",
                                    z="Heart Attack Risk", nbinsx=10, nbinsy=10,
                                    histfunc="avg", color_continuous_scale="RdBu_r",
                                    title="🏃‍♂️ Stress vs Exercise vs Risk")
    st.plotly_chart(fig_stress)

    # Medication usage
    med_usage = df.groupby("Heart Attack Risk (Text)")["Medication Use"].mean().reset_index()
    med_usage["Medication Use %"] = (med_usage["Medication Use"] * 100).round(1)
    st.plotly_chart(px.bar(med_usage, x="Heart Attack Risk (Text)", y="Medication Use %",
                           color="Heart Attack Risk (Text)", text="Medication Use %",
                           title="💊 Medication Usage by Risk Class"))


# ------------------------- DATASET UTILITY ------------------------- #
def explain_dashboard():
    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})
    bp = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic blood pressure'] = pd.to_numeric(bp[0])
    df['Diastolic blood pressure'] = pd.to_numeric(bp[1])
    df.drop(columns=["Patient ID", "Country", "Continent", "Hemisphere", "Blood Pressure"], inplace=True)
    df.dropna(inplace=True)
    return df[selected_variables]


