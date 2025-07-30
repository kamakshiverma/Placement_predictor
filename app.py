# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
log_model = joblib.load('placement_model.pkl')
rf_model = joblib.load('rf_placement_model.pkl')

# Set page config
st.set_page_config(page_title="Student Placement Predictor", layout="centered")

st.title("🎓 Smart Student Placement Predictor")
st.markdown("Predict if a student is likely to be placed based on academic and personal attributes.")

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

# Input fields
st.subheader("📥 Enter Student Details")

iq = st.number_input("IQ", min_value=50, max_value=200, step=1)
prev_result = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
academic_perf = st.number_input("Academic Performance", min_value=0, max_value=10, step=1)
internship = st.selectbox("Internship Experience", ["No", "Yes"])
extra_curricular = st.number_input("Extra Curricular Score", min_value=0, max_value=10, step=1)
communication = st.number_input("Communication Skills", min_value=0, max_value=10, step=1)
projects = st.number_input("Number of Projects Completed", min_value=0, max_value=10, step=1)

# Convert categorical input
internship_val = 1 if internship == "Yes" else 0

input_data = np.array([[iq, prev_result, cgpa, academic_perf, internship_val, extra_curricular, communication, projects]])

# Predict button
if st.button("🎯 Predict Placement"):
    if model_choice == "Logistic Regression":
        model = log_model
    else:
        model = rf_model

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"✅ Student is likely to be **Placed**! \n\n📊 Confidence: `{proba*100:.2f}%`")
    else:
        st.error(f"❌ Student is **Not Likely to be Placed**.\n\n📊 Confidence: `{(1-proba)*100:.2f}%`")

# Optional: Feature Importance for RF
if model_choice == "Random Forest":
    st.subheader("📌 Feature Importance (Random Forest)")
    importance = rf_model.feature_importances_
    features = ['IQ', 'Prev Result', 'CGPA', 'Academic', 'Internship', 'Extra Curricular', 'Communication', 'Projects']
    df_feat = pd.DataFrame({'Feature': features, 'Importance': importance})
    df_feat = df_feat.sort_values(by='Importance', ascending=False)

    st.bar_chart(df_feat.set_index("Feature"))
