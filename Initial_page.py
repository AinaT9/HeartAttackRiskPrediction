import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>Heart Attack Risk Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.header("What is your risk of heart attack?")
st.sidebar.title("Menu")
profile = st.sidebar.selectbox("Profile",["Patient","Doctor", "Data Analyst"])


# Primera fila
col1, col2, col3 = st.columns(3)
with col1:
    sex = st.segmented_control("Sex", ["M", "F"])
with col2:
    age = st.text_input("Age")
with col3:
    option = st.checkbox("Option 1")

# Segunda fila
col4, col5, col6 = st.columns(3)
with col4:
    option2 = st.checkbox("Option 2")
with col5:
    option3 = st.number_input("Numeric option")
with col6:
    option4 = st.number_input("Numeric option 2")

# Tercera fila (puedes agregar más si lo necesitas)
col7, col8, col9 = st.columns(3)
with col7:
    st.write("Espacio disponible")
with col8:
    st.write("Puedes agregar más widgets")
with col9:
    st.write("O dejar vacío")

_, _, col12 = st.columns(3)

with col12: 
    b = st.button("CALCULATE!")

if b:
    st.subheader("Your risk is")
    value = random.randint(0,100)
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

