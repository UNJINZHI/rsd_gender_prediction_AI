import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

# Prepare Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

# --- 2. MAIN PAGE LAYOUT ---
st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

# --- SECTION 1: INPUT PATIENT DATA (Main Page) ---
st.subheader("📝 Enter Patient Details")

# We use columns to organize inputs neatly instead of a long list
col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Patient Full Name", "John Doe")
    age = st.number_input('Age', 29, 90, 54)
    sex = st.selectbox('Sex', (1, 0), format_func=lambda x: 'Male' if x == 1 else 'Female')
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 130)
    chol = st.number_input('Cholesterol (mg/dl)', 100, 600, 246)

with col2:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (1, 0), format_func=lambda x: 'True' if x == 1 else 'False')
    restecg = st.selectbox('Resting ECG', (0, 1, 2), format_func=lambda x: ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x])
    thalach = st.number_input('Max Heart Rate', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.number_input('ST Depression (Oldpeak)', 0.0, 10.0, 1.0, step=0.1)

with col3:
    slope = st.selectbox('Slope of Peak Exercise ST', (0, 1, 2), format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    ca = st.slider('Major Vessels (0-3)', 0, 4, 0)
    thal = st.selectbox('Thalassemia', (0, 1, 2, 3), format_func=lambda x: ['Null', 'Fixed Defect', 'Normal', 'Reversable Defect'][x])
    cp = st.selectbox('Chest Pain Type', (0, 1, 2, 3), format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'][x])

st.markdown("---")

# Prepare the data dictionary for the model
input_data = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
input_df = pd.DataFrame(input_data, index=[0])

# --- SECTION 2: PATIENT REPORT ---
st.subheader(f"📋 Medical Report: {name}")

# Create a readable summary table
report_data = {
    'Attribute': ['Age', 'Sex', 'Chest Pain', 'BP', 'Cholesterol', 'Max HR', 'Exercise Angina', 'ST Depression'],
    'Value': [
        f"{age} years",
        'Male' if sex == 1 else 'Female',
        ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'][cp],
        f"{trestbps} mm Hg",
        f"{chol} mg/dl",
        f"{thalach} bpm",
        'Yes' if exang == 1 else 'No',
        oldpeak
    ]
}
report_df = pd.DataFrame(report_data)

# Display report as a clean table (Transposed for better look if needed, but standard is fine)
st.table(report_df.set_index('Attribute').T)

# --- SECTION 3: DIAGNOSIS ---
st.subheader("🔍 Diagnosis Results")

if st.button("Analyze Patient Data"):
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = rfc.predict(input_scaled)
    
    # Display Result (No Confidence Score)
    if prediction[0] == 1:
        st.error(f"⚠️ **POSITIVE**: The model predicts high likelihood of Heart Disease.")
        st.write("Suggested Action: Immediate consultation recommended.")
    else:
        st.success(f"✅ **NEGATIVE**: The model predicts the patient is healthy.")
        st.write("Suggested Action: Routine checkup recommended.")
