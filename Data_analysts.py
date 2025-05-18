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
import plotly.express as px
import altair as alt
import pycountry
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split



modelo_cargado = joblib.load("Modelo/modelo_rf.joblib")

columns = [
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
Feature_categories = {
    "Age": "Demographic",
    "Gender": "Demographic",
    "Country": "Demographic",

    "Cholesterol": "Clinical",
    "Heart Rate": "Clinical",
    "Diabetes": "Clinical",
    "Family History": "Clinical",
    "Previous Heart Problems": "Clinical",
    "Medication Use": "Clinical",
    "Systolic blood pressure": "Clinical",
    "Diastolic blood pressure": "Clinical",
    "BMI": "Clinical",
    "Triglycerides": "Clinical",

    "Smoking": "Lifestyle",
    "Obesity": "Lifestyle",
    "Alcohol Consumption": "Lifestyle",
    "Exercise Hours Per Week": "Lifestyle",
    "Diet": "Lifestyle",
    "Stress Level": "Lifestyle",
    "Sedentary Hours Per Day": "Lifestyle",
    "Sleep Hours Per Day": "Lifestyle"
}

color_theme_map = {
    'blues': px.colors.sequential.Blues,
    'cividis': px.colors.sequential.Cividis,
    'greens': px.colors.sequential.Greens,
    'inferno': px.colors.sequential.Inferno,
    'magma': px.colors.sequential.Magma,
    'plasma': px.colors.sequential.Plasma,
    'reds': px.colors.sequential.Reds,
    'viridis': px.colors.sequential.Viridis,
    'turbo': px.colors.sequential.Turbo,
    'rainbow': px.colors.qualitative.Bold 
}


def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=25, cornerRadius=15).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=90, height=90)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=18,
                          fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=25, cornerRadius=10).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=90, height=90)
    return plot_bg + plot + text


def get_iso3(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except:
        return None


def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(
        input_df,
        locations=input_id,
        color=input_column,
        locationmode="ISO-3",
        color_continuous_scale=input_color_theme,
        range_color=(0, 100),
        scope="world",
        labels={input_column: input_column}
    )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )
    return choropleth
def plot_by_variable(name, df, color):
    summary = []
    for group in df[name].unique():
        subset = df[df[name] == group]
        for risk in [0, 1]:
            real_count = (subset['Heart Attack Risk'] == risk).sum()
            pred_count = (subset['Predicted_Class'] == risk).sum()

            summary.extend([
                {'Group': group, 'Risk': 'Low Risk' if risk == 0 else 'High Risk', 'Type': 'Real', 'Count': real_count},
                {'Group': group, 'Risk': 'Low Risk' if risk == 0 else 'High Risk', 'Type': 'Predicted', 'Count': pred_count},
            ])

    summary_df = pd.DataFrame(summary)
    fig = px.bar(
        summary_df,
        x='Group',
        y='Count',
        color='Type',
        barmode='group',
        facet_col='Risk',
        color_discrete_sequence=color_theme_map[color],
        labels={'Group': name, 'Count': 'Número de Casos'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_dashboard(selected_color_theme):
    st.header("Data Scientist Dashboard: Model Testing Interface")
    col = st.columns((1.5, 3, 3), gap='medium')
    with col[0]:
        st.markdown('#### Model Metrics ')
        st.markdown('##### Accuracy ')
        st.altair_chart(make_donut(75, 'Accuracy', 'green'),
                        use_container_width=True)
        st.markdown('##### Precision on High Risk class ')
        st.altair_chart(make_donut(75, 'Accuracy', 'blue'),
                        use_container_width=True)
        st.markdown('##### Recall on High Risk class ')
        st.altair_chart(make_donut(75, 'Accuracy', 'orange'),
                        use_container_width=True)
        st.markdown('##### Specificity on Low Risk class ')
        st.altair_chart(make_donut(75, 'Accuracy', 'red'),
                        use_container_width=True)
        st.markdown('##### F1-SCORE ')
        st.altair_chart(make_donut(75, 'Accuracy', 'green'),
                        use_container_width=True)
        st.markdown('#### Dataset Statistics')
        st.metric(label="'%' of Predicted High Risk", value=10, border=True)
        st.metric(label="Average Probability of class High Risk",
                  value=10, border=True)

    with col[1]:
        st.markdown('#### Risk by Country')
        df = get_df()
        df['country_code'] = df['Country'].apply(get_iso3)
        risk_class = st.selectbox("Select Risk Class to Show", options=[
                                  "High Risk", "Low Risk"])
        agg_df = df.groupby('country_code')[
            'Heart Attack Risk'].mean().reset_index()
        if risk_class == "High Risk":
            agg_df['High Risk %'] = agg_df['Heart Attack Risk'] * 100
            class_risk = 'High Risk %'
        else:
            agg_df['Low Risk %'] = (1 - agg_df['Heart Attack Risk']) * 100
            class_risk = 'Low Risk %'

        choropleth = make_choropleth(
            agg_df, 'country_code', class_risk, selected_color_theme)
        st.plotly_chart(choropleth, use_container_width=True)

        st.subheader("How Features Influence Predictions")
        st.markdown("""
        <p style="margin-bottom:4px;">The plot below uses <b>SHAP values</b> to show how each feature affects the model's prediction.</p>
        <p style="margin-bottom:4px;">Each dot is a person in the dataset.</p>
        <p style="margin-bottom:4px;">The horizontal position shows whether the feature increases (right) or decreases (left) the risk.</p>
        <p style="margin-bottom:4px;">The color represents the actual value of the feature (e.g., high or low cholesterol).</p>
        <p style="margin-bottom:4px;">Features are sorted by overall importance.</p>
        """, unsafe_allow_html=True)
        explainer, X, shap_values = explain_dashboard()
        st_shap(shap.plots.beeswarm(shap_values, max_display=15), height=500, width=900)
        

        with col[2]:
            st.markdown('#### Prediction vs Real')
            selected_categories2 = st.selectbox(
                label="Filter by Feature Category",
                options=sorted(set(Feature_categories.keys())),
            )
            n = 200
            smoking_status = np.random.choice(['Non-Smoker', 'Heavy Smoker'], size=n)
            true_class = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
            predicted_class = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
            df = pd.DataFrame({
                'Smoking': smoking_status,
                'Heart Attack Risk': true_class,
                'Predicted_Class': predicted_class
            })
            plot_by_variable("Smoking", df, selected_color_theme)

            feature_importance = modelo_cargado.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            importance_df['Category'] = importance_df['Feature'].map(Feature_categories)
            with st.expander("🔧 Filter Options"):
                selected_categories = st.multiselect(
                    "Filter by Feature Category",
                    options=sorted(set(Feature_categories.values())),
                    default=sorted(set(Feature_categories.values()))
                )
                filtered_df = importance_df.copy()
                filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
            if filtered_df.empty:
                st.warning("No features match your filters.")
            else:
                fig = px.bar(
                    filtered_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Random Forest)',
                    color='Importance',
                    color_continuous_scale=selected_color_theme
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            


        #st_shap(shap.plots.scatter(shap_values[:, 'Cholesterol']))

    # st.subheader("Enter patient data for ML model evaluation")
    # inputs = []
    # # for i, col in enumerate(columns):
    # #     if col in ['Gender', 'Diabetes', 'Smoking', 'Alcohol Consumption', 'Obesity']:
    # #         val = st.radio(f"{col}", ["Yes", "No"])
    # #         val = 1 if val == "Yes" else 0
    # #     else:
    # #         val = st.slider(f"{col}", min_val, max_val, int((min_val + max_val) / 2))
    # #     inputs.append(val)

    # if st.button("Predict and Explain"):
    #     st.subheader("Predicted Heart Attack Risk:")
    #     risk = get_prediction(inputs)
    #     display_risk_indicator(risk)

    #     # SHAP
    #     inputs_arr = np.array(inputs).reshape(1, -1)
    #     explainer = shap.Explainer(modelo_cargado)
    #     shap_values = explainer(inputs_arr)
    #     shap_value = shap.Explanation(
    #         values=shap_values.values[:, 1],
    #         base_values=shap_values.base_values[0][1] * 100,
    #         data=shap_values.data,
    #         feature_names=columns
    #     )
    #     st.write("### SHAP Explanation (Model Interpretation)")
    #     st_shap(shap.plots.force(shap_value, matplotlib=True))
    #     plt.show()


def get_df():
    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.drop(columns=['Patient ID', 'Income',
                 'Physical Activity Days Per Week', 'Continent', 'Hemisphere'])
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
    df['Diet'] = df['Diet'].map(diet_map)
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic blood pressure'] = pd.to_numeric(bp_split[0])
    df['Diastolic blood pressure'] = pd.to_numeric(bp_split[1])
    df = df.drop(columns=['Blood Pressure'])
    return df

def get_metrics():
    df = get_df()
    target_column = "Heart Attack Risk"
    X = df.drop(columns=[target_column, "Heart Attack Risk"])
    y = df[target_column]
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

@st.cache_resource
def explain_dashboard():
    df = get_df()
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    X = df[columns]
    explainer = shap.Explainer(modelo_cargado)
    X = X.sample(n=200, random_state=42)
    shap_values = explainer(X)[:,:,1]
    return explainer, X, shap_values


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
