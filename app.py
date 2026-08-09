import streamlit as st
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph,Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import shap
import matplotlib.pyplot as plt

def create_pdf(report):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>Stroke Prediction Patient Report</b>", styles["Title"]))
    story.append(Spacer(1,12))

    story.append(Paragraph(report.replace("\n","<br/>"), styles["BodyText"]))

    doc.build(story)

    buffer.seek(0)

    return buffer



# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("stroke_xgboost_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, model_features = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🧠 First Time Stroke Prediction Using Machine Learning and Patient Segmentation")
st.write("Doctor Decision Support System")

st.divider()

# -------------------------------
# Doctor Dashboard
# -------------------------------
st.header("👨‍⚕️ Doctor Dashboard")
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    glucose = st.number_input("Average Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    smoking = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes"]
    )
    work = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "children", "Never_worked"]
    )

if st.button("🔍 Predict Stroke Risk"):

    input_df = pd.DataFrame({
        "age":[age],
        "hypertension":[hypertension],
        "heart_disease":[heart_disease],
        "avg_glucose_level":[glucose],
        "bmi":[bmi],
        "gender_Male":[1 if gender=="Male" else 0],
        "smoking_status_smokes":[1 if smoking=="smokes" else 0],
        "smoking_status_formerly smoked":[1 if smoking=="formerly smoked" else 0],
        "work_type_Private":[1 if work=="Private" else 0],
        "work_type_Self-employed":[1 if work=="Self-employed" else 0],
        "work_type_children":[1 if work=="children" else 0],
        "work_type_Never_worked":[1 if work=="Never_worked" else 0],
    })

    # Match model features
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk_percentage = probability * 100

    st.divider()
    st.subheader("Stroke Risk Result")

    if risk_percentage >= 50:
        st.error("🔴 High Stroke Risk")
        risk_level = "High Risk"
    elif risk_percentage >= 20:
        st.warning("🟡 Medium Stroke Risk")
        risk_level = "Medium Risk"
    else:
        st.success("🟢 Low Stroke Risk")
        risk_level = "Low Risk"

    st.metric(
        "Stroke Probability",
        f"{risk_percentage:.2f}%"
    )

    st.divider()

    st.subheader("👤 Patient Segmentation")

    if age >= 60 or hypertension == 1 or heart_disease == 1:
        patient_segment = "High Risk Patient"
    elif age >= 40 or bmi >= 25:
        patient_segment = "Medium Risk Patient"
    else:
        patient_segment = "Low Risk Patient"

    st.info(patient_segment)

    st.divider()

    st.subheader("⚠️ Risk Factor Summary")

    if age >= 60:
        st.write("⚠️ Older age")

    if hypertension == 1:
        st.write("⚠️ Hypertension")

    if heart_disease == 1:
        st.write("⚠️ Heart Disease")

    if glucose > 150:
        st.write("⚠️ High Glucose")

    if bmi >= 25:
        st.write("⚠️ High BMI")

    if smoking != "never smoked":
        st.write("⚠️ Smoking History")

# ===============================
    # PDF REPORT
    # ===============================

    st.divider()

    st.subheader("📄 Patient Report")

    report = f"""
    Stroke Prediction Report

    Age: {age}
    Gender: {gender}
    BMI: {bmi}
    Glucose: {glucose}

    Risk Level: {risk_level}
    Stroke Probability: {risk_percentage:.2f}%

    Patient Segment: {patient_segment}
    """

    pdf = create_pdf(report)

    st.download_button(
        label="⬇️ Download Patient PDF Report",
        data=pdf,
        file_name="stroke_patient_report.pdf",
        mime="application/pdf"
    )



  # ===============================
  # SHAP EXPLAINABLE AI
  # ===============================
st.divider()

st.subheader("🔍 SHAP Explainable AI")

try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig = plt.figure(figsize=(8,5))

        shap.summary_plot(
            shap_values,
            input_df,
            plot_type="bar",
            show=False
        )

        st.pyplot(fig)

except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")


