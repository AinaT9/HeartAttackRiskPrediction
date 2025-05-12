import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
import Doctors_dashboard
import Patient_dashboard
import Data_analysts

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>Heart Attack Risk Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.sidebar.title("Menu")
profile = st.sidebar.selectbox("Profile",["Patient","Doctor", "Data Analyst"])
if profile == "Doctor":
    Doctors_dashboard.show_dashboard()
elif profile =="Patient":
    Patient_dashboard.show_dashboard()
else: 
    Data_analysts.show_dashboard()
