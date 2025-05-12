import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image

def show_dashboard():
    st.header("Doctor's Dashboard")
    st.write("Here is the doctor's profile and appointments.")