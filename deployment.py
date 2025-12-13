import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Title and Layout
st.set_page_config(page_title="Heart Health App", layout="wide")
st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient data below to get a risk assessment.")
st.markdown("---")

# 2. Load Data and Train Model
# We use this special command to stop the app from reloading the data every time you click a button
@st.cache_data
def train_model():
    # Load data
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data (Make numbers roughly the same size)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Run the function above
model, scaler = train_model()

# 3. Create the Input Form
st.header("📝 Patient Information")

# We create 3 columns to make it look organized
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. General Info")
    name = st.text_input("Patient Name", "John Doe")
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    
    # Simple way to handle text options:
    sex_option = st.selectbox("Sex", ["Male", "Female"])
    # Convert text back to number for the model (Male=1, Female=0)
    if sex_option == "Male":
        sex = 1
    else:
        sex = 0

with col2:
    st.subheader("2. Vitals")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs_option = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])
    if fbs_option == "Yes":
        fbs = 1
    else:
        fbs = 0
    thalach = st.number_input("Max Heart Rate", value=150)

with col3:
    st.subheader("3. Heart Exam")
    cp_option = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    # Convert label to number (0-3)
    if cp_option == "Typical Angina": cp = 0
    elif cp_option == "Atypical Angina": cp = 1
    elif cp_option == "Non-anginal Pain": cp = 2
    else: cp = 3
        
    exang_option = st.selectbox("Pain during Exercise?", ["No", "Yes"])
    if exang_option == "Yes":
        exang = 1
    else:
        exang = 0
        
    oldpeak = st.number_input("ST Depression", value=1.0)

# Extra inputs (Put them below to keep columns clean)
st.markdown("---")
st.subheader("4. Advanced Tests (Optional defaults set)")
c1, c2, c3 = st.columns(3)

with c1:
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
with c2:
    slope = st.selectbox("ST Slope", [0, 1, 2])
with c3:
    ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# 4. Show Summary
st.markdown("---")
st.header(f"📊 Summary for {name}")

# Use 'metrics' to show big numbers nicely
m1, m2, m3 = st.columns(3)
m1.metric("Blood Pressure", f"{trestbps} mm Hg")
m2.metric("Cholesterol", f"{chol} mg/dl")
m3.metric("Max Heart Rate", f"{thalach} bpm")

# 5. Prediction Logic
if st.button("Analyze Risk Now"):
    # Prepare the data exactly how the model expects it
    user_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    # Scale the data
    user_data_scaled = scaler.transform(user_data)
    
    # Get prediction
    prediction = model.predict(user_data_scaled)
    
    # Show result
    if prediction[0] == 1:
        st.error("⚠️ HIGH RISK: The model predicts potential Heart Disease.")
        st.write("Please consult a doctor.")
    else:
        st.success("✅ LOW RISK: The model predicts a Healthy Heart.")
        st.write("Keep maintaining a healthy lifestyle!")
