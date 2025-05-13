#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import joblib

# %%
df = pd.read_csv("Modelo/data/heart-attack-risk-prediction-dataset.csv")
df.head()

# %%
df.columns

# %%
df.describe()

# %%
df['Heart Attack Risk (Binary)'].value_counts()
# %%
df.info()

# %%
df.isna().sum()

# %%
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# %%
imputer = KNNImputer(n_neighbors=2)

df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(df)
# %%
df.isna().sum()

#%%
df.head()
# %%
correlation_matrix = df.corr()
correlation_matrix['Heart Attack Risk (Text)']

# %%
target_corr = correlation_matrix['Heart Attack Risk (Binary)'].drop('Heart Attack Risk (Binary)')

target_corr_sorted = target_corr.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr_sorted.values, y=target_corr_sorted.index, palette='coolwarm')
plt.title("Correlación con la variable 'Hearth-attack'")
plt.xlabel("Coeficiente de correlación")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# %%
target_column = "Heart Attack Risk (Binary)"
X = df.drop(columns=[target_column, "Heart Attack Risk (Text)"])
y = df[target_column]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, y_train.shape

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# %%
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# %%
# Guardar el modelo
joblib.dump(rf, 'Modelo/modelo_rf.joblib')

#%%
# Cargar el modelo
modelo_cargado = joblib.load('Modelo/modelo_rf.joblib')

# %%
predicciones = modelo_cargado.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Precisión: {accuracy}')
print(classification_report(y_test, predicciones))
# %%

selected_variables = [
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
X = df[selected_variables]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, y_train.shape

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# %%
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest Patients")
plt.show()

# %%
# Guardar el modelo
joblib.dump(rf, 'Modelo/modelo_rf_patients.joblib')

#%%
# Cargar el modelo
modelo_cargado = joblib.load('Modelo/modelo_rf_patients.joblib')

# %%
predicciones = modelo_cargado.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Precisión: {accuracy}')
print(classification_report(y_test, predicciones))