# 🫀 Heart Attack Risk Prediction Dashboard

This web application helps users assess their risk of a heart attack using key health indicators. It’s built with Streamlit, offering a simple, interactive way to visualize and predict potential risk levels.

## 🚀 Getting Started
### 1. Online 
👉 Access the dashboard directly through this link:
🔗 [Heart Attack Risk Prediction App](https://heartattackriskpredictiondashboard.streamlit.app/)

### 2. Run Locally
If you prefer to run the app on your own machine:

🔧 Prerequisites:
Make sure you have Python 3.9 and execute the following code:
``` bash
pip install requirements.txt 
```
In your terminal or command prompt, navigate to the project directory and execute:
``` bash
streamlit run Initial_page.py
```
The app will launch in your default web browser.

## 🧠 Features
- Dashboard Responsive across different screen sizes. 
- incorporates distinct interactive visualizations to illustrate various aspects 
of the XAI methods used.
- include appropriate navigation and help features (sidebar menu, text explanations,tooltips,instructions, etc)
- The code is commented and documented.

📁 Project Structure
```bash 
.
├── Initial_page.py         # Main Streamlit app file
├── Data_analysts.py        # Data Analysts dashboard
├── Doctors_dashboard.py    # Doctors dashboard
├── Patient_dashboard.py    # Patient dashboard
├── Modelo/                 # Folder containing ML models 
    ├── data/               # Folder containing the dataset
    ├── data-analysis.py    # File containing the data analyisis, training and  validation of models      
├── images/                   # Folder containing images 
├── requirements.txt        # Dependencies (optional)
└── README.md               # Project documentation
```
