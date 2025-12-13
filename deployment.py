import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediHeart Pro | AI Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Modern Medical Look) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2c3e50;
    }
    /* Header Styling */
    h1 {
        color: #0f4c81;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #4b6584;
    }
    /* Custom Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4b7bec, #3867d6);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        border-radius: 8px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #3867d6, #2d55aa);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIC & MODEL TRAINING ---
@st.cache_data
def train_model():
    # Load Data
    try:
        df = pd.read_csv('heart.csv').drop_duplicates()
        
        # Define Features & Target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split & Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        return model, X_train.columns
    except FileNotFoundError:
        st.error("🚨 Error: 'heart.csv' file not found. Please upload it to your repository.")
        st.stop()

model, feature_names = train_model()

# --- 4. SIDEBAR (PATIENT DATA INPUT) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60)
    st.title("Patient Data")
    st.markdown("---")
    
    # 1. Demographics
    st.subheader("👤 Profile")
    name = st.text_input("Full Name", "John Doe")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 20, 100, 50)
    sex = c2.selectbox("Sex", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.markdown("---")
    
    # 2. Vitals
    st.subheader("📊 Vitals")
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    fbs = st.selectbox("Fasting Sugar > 120?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("---")
    
    # 3. Clinical Exams
    st.subheader("🔬 Clinical Exams")
    cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3), 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    exang = st.selectbox("Exercise Angina?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
    
    with st.expander("Advanced EKG/Stresstest"):
        restecg = st.selectbox("Resting ECG", (0, 1, 2), format_func=lambda x: ["Normal", "ST Abnormality", "LV Hypertrophy"][x])
        slope = st.selectbox("ST Slope", (2, 1, 0), format_func=lambda x: {2:"Upsloping", 1:"Flat", 0:"Downsloping"}.get(x))
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1)
        ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", (2, 1, 3), format_func=lambda x: {2:"Normal", 1:"Fixed Defect", 3:"Reversible Defect"}.get(x))

    # Prepare Input Data
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)

# --- 5. MAIN DASHBOARD ---
st.title("🩺 MediHeart Pro Dashboard")
st.markdown(f"**Patient:** {name} | **ID:** #82910 | **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")

# Vitals Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Blood Pressure", f"{trestbps}", "mm Hg")
col2.metric("Cholesterol", f"{chol}", "mg/dl", delta_color="inverse")
col3.metric("Max HR", f"{thalach}", "bpm")
col4.metric("ST Depression", f"{oldpeak:.1f}")

st.markdown("---")

# Prediction Section
st.subheader("🤖 AI Risk Assessment")

if st.button("Analyze Patient Data"):
    with st.spinner("Processing Vitals & EKG..."):
        # Prediction Logic
        # Class 0 = Disease, Class 1 = Healthy
        # We want probability of Class 0
        prob_disease = model.predict_proba(input_df)[0][0]
        prob_healthy = model.predict_proba(input_df)[0][1]
        
        # VISUALIZATION
        col_res, col_chart = st.columns([2, 1])
        
        with col_res:
            if prob_disease > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
                st.markdown(f"""
                ### Probability of Heart Disease: <span style='color:#d63031'>{prob_disease*100:.1f}%</span>
                The model has detected patterns consistent with cardiovascular disease.
                
                **Primary Contributing Factors (General):**
                * Check for Exercise Induced Angina
                * Review ST Depression levels ({oldpeak})
                * Check Major Vessel count ({ca})
                """, unsafe_allow_html=True)
            else:
                st.success("✅ LOW RISK / HEALTHY")
                st.markdown(f"""
                ### Probability of Heart Disease: <span style='color:#27ae60'>{prob_disease*100:.1f}%</span>
                The patient shows a healthy heart profile.
                
                **Recommendations:**
                * Continue routine checkups.
                * Maintain current diet and exercise.
                """, unsafe_allow_html=True)

        with col_chart:
            st.write("### Risk Probability")
            # Create a simple progress bar chart
            st.progress(int(prob_disease * 100))
            st.caption("0% = Healthy | 100% = Critical")

else:
    st.info("👈 Please adjust patient details in the Sidebar and click **Analyze**.")
