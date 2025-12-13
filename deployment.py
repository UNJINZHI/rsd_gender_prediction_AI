import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

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

# Train Model (Random Forest is usually the best single model)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

# --- 2. SIDEBAR (User Inputs) ---
st.sidebar.header("Patient Information")

def user_input_features():
    # Added Patient Name Field
    name = st.sidebar.text_input("Patient Full Name", "John Doe")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Medical Details")
    
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', (1, 0), format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3), 
                              format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 130)
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (1, 0), format_func=lambda x: 'True' if x == 1 else 'False')
    restecg = st.sidebar.selectbox('Resting ECG', (0, 1, 2), 
                                   format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
    thalach = st.sidebar.slider('Max Heart Rate', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', (0, 1, 2),
                                 format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    ca = st.sidebar.slider('Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3), 
                                format_func=lambda x: ['Null', 'Fixed Defect', 'Normal', 'Reversable Defect'][x])

    # Store raw values for prediction
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # Create readable values for the report
    readable_data = {
        'Patient Name': name,
        'Age': age,
        'Sex': 'Male' if sex == 1 else 'Female',
        'Chest Pain Type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][cp],
        'Resting Blood Pressure': f"{trestbps} mm Hg",
        'Cholesterol': f"{chol} mg/dl",
        'Fasting Blood Sugar > 120': 'Yes' if fbs == 1 else 'No',
        'Resting ECG': ['Normal', 'ST-T Wave Abnormality', 'LV Hypertrophy'][restecg],
        'Max Heart Rate': thalach,
        'Exercise Angina': 'Yes' if exang == 1 else 'No',
        'ST Depression': oldpeak,
        'ST Slope': ['Upsloping', 'Flat', 'Downsloping'][slope],
        'Major Vessels': ca,
        'Thalassemia': ['Null', 'Fixed Defect', 'Normal', 'Reversable Defect'][thal]
    }
    
    return pd.DataFrame(data, index=[0]), readable_data

input_df, report_data = user_input_features()

# --- 3. MAIN PAGE ---
st.title("❤️ Heart Disease Prediction System")
st.write("### Patient Medical Report")

# Display the Readable Report
# Convert dictionary to a clean DataFrame for display
report_df = pd.DataFrame(list(report_data.items()), columns=['Attribute', 'Value'])
st.table(report_df)

# --- 4. PREDICTION LOGIC ---
st.subheader("Diagnosis Results")

# Prediction Button
if st.button("Analyze & Predict"):
    # Scale the user input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = rfc.predict(input_scaled)
    prob = rfc.predict_proba(input_scaled)

    # Logic for display
    if prediction[0] == 1:
        st.error(f"⚠️ **POSITIVE**: High likelihood of Heart Disease detected.")
        st.write(f"**Confidence Score:** {prob[0][1] * 100:.2f}%")
        st.write("Suggested Action: Consult a cardiologist immediately for further testing.")
    else:
        st.success(f"✅ **NEGATIVE**: Patient appears healthy.")
        st.write(f"**Confidence Score:** {prob[0][0] * 100:.2f}%")
        st.write("Suggested Action: Maintain a healthy lifestyle and regular checkups.")
