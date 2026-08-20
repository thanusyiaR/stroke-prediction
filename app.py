"""Revised Streamlit prototype using the leakage-controlled training pipeline."""
from io import BytesIO
from pathlib import Path
from dashboard_research_evidence import render_research_evidence
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "revised_outputs"

st.set_page_config(page_title="Stroke Risk Prediction Using Machine Learning and Patient Segmentation", page_icon="🧠", layout="wide")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(OUTPUT / "stroke_xgboost_pipeline.joblib")
    metadata = json.loads((OUTPUT / "model_metadata.json").read_text())
    results = pd.read_csv(OUTPUT / "independent_test_results.csv")
    thresholds = pd.read_csv(OUTPUT / "threshold_analysis.csv")
    imbalance = pd.read_csv(OUTPUT / "xgboost_imbalance_strategy_comparison.csv")
    features = pd.read_csv(OUTPUT / "feature_manifest.csv")
    return pipeline, metadata, results, thresholds, imbalance, features


def clinical_segment(patient):
    factors = []
    if patient["age"] >= 60: factors.append("Age 60 years or older")
    if patient["hypertension"] == 1: factors.append("Hypertension")
    if patient["heart_disease"] == 1: factors.append("Heart disease")
    if patient["avg_glucose_level"] >= 200: factors.append("Average glucose of 200 mg/dL or higher")
    if patient["bmi"] >= 30: factors.append("BMI of 30 kg/m² or higher")
    score = len(factors)
    label = "Low" if score <= 1 else "Medium" if score == 2 else "High"
    return score, label, factors


def make_pdf(patient, probability, classification, segment, factors):
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    story = [Paragraph("Stroke Risk Prediction Using Machine Learning and Patient Segmentation - Patient Report", styles["Title"]), Spacer(1, 12)]
    rows = [["Field", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in patient.items()]
    rows += [["Model probability", f"{probability:.2%}"], ["Threshold classification", classification],
             ["Project-specific risk-factor segment", segment], ["Detected factors", ", ".join(factors) or "None"]]
    table = Table(rows, colWidths=[190, 300])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [table, Spacer(1, 12), Paragraph(
        "Academic decision-support prototype only. This output is not a diagnosis, a validated clinical risk score, or a substitute for professional assessment.",
        styles["BodyText"])]
    SimpleDocTemplate(buffer).build(story)
    buffer.seek(0)
    return buffer


pipeline, metadata, results, thresholds, imbalance, features = load_artifacts()
threshold = float(metadata["classification_threshold"])

st.title("🧠 Stroke Risk Prediction Using Machine Learning and Patient Segmentation")
st.caption("Revised academic decision-support prototype | XGBoost with training-derived class weighting")
st.warning("For academic demonstration only. The model has not been externally or clinically validated.")

with st.expander("Leakage-controlled preprocessing and feature scaling"):
    st.markdown(
        """
        - **Numeric predictors:** age, average glucose, and BMI are median-imputed and standardized using `StandardScaler`.
        - **Binary predictors:** hypertension and heart disease pass through unchanged as 0/1 values.
        - **Categorical predictors:** gender, marital status, work type, residence type, and smoking status are most-frequent imputed and one-hot encoded.
        - **Leakage control:** every imputer, scaler, and encoder is fitted only on the relevant training fold during five-fold stratified cross-validation. The independent test set and new patient inputs are transformed using the already-fitted pipeline.
        - **KNN relevance:** standardization prevents variables with larger numerical ranges, particularly glucose, from dominating distance calculations.
        """
    )

with st.expander("Complete model-feature list and leakage exclusions"):
    st.write(
        "The revised model uses 10 original eligible predictors, transformed into "
        "16 model features. Identifier, target, risk score, patient segment, and all "
        "other segmentation-derived labels are excluded from predictive training."
    )
    st.dataframe(features, width="stretch", hide_index=True)

with st.expander("Basis and limitations of patient segmentation"):
    st.markdown(
        """
        The project-specific framework assigns one equal point for each condition:
        age ≥60, BMI ≥30 kg/m², hypertension, heart disease, and average glucose
        ≥200 mg/dL. Scores 0–1 are labelled Low, 2 Medium, and 3–5 High.

        BMI ≥30 follows the WHO adult obesity boundary. The glucose threshold is
        used only as a high-glucose flag informed by established diabetes-testing
        thresholds; the dataset variable is not sufficient to diagnose diabetes.
        Hypertension, heart disease, obesity, and diabetes-related hyperglycaemia
        are recognized stroke-risk factors. Age ≥60 is a pragmatic project boundary.
        Equal weighting supports transparency and does not imply equal clinical effect.

        **This framework was not clinically derived, calibrated, or validated and is
        not a stroke-risk score. It must not guide diagnosis or treatment.**
        """
    )

with st.form("patient_form"):
    left, middle, right = st.columns(3)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", 0.08, 100.0, 55.0, 1.0)
        hypertension = int(st.selectbox("Hypertension", ["No", "Yes"]) == "Yes")
        heart_disease = int(st.selectbox("Heart disease", ["No", "Yes"]) == "Yes")
    with middle:
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence type", ["Urban", "Rural"])
    with right:
        glucose = st.number_input("Average glucose (mg/dL)", 40.0, 350.0, 100.0, 1.0)
        bmi = st.number_input("BMI (kg/m²)", 10.0, 80.0, 25.0, 0.1)
        smoking = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Evaluate patient")

if submitted:
    patient = {"gender": gender, "age": age, "hypertension": hypertension,
               "heart_disease": heart_disease, "ever_married": ever_married,
               "work_type": work_type, "Residence_type": residence,
               "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking}
    frame = pd.DataFrame([patient])
    probability = float(pipeline.predict_proba(frame)[0, 1])
    classification = "Stroke-positive flag" if probability >= threshold else "No stroke-positive flag"
    score, segment, factors = clinical_segment(patient)

    a, b, c = st.columns(3)
    a.metric("Predicted probability", f"{probability:.2%}")
    b.metric(f"Classification at {threshold:.2f}", classification)
    c.metric("Risk-factor segment", f"{segment} ({score}/5)")
    st.caption("The probability is a model output. The equally weighted segment is a separate project-specific descriptive summary—not a clinically validated stroke-risk score.")

    st.subheader("Detected segmentation factors")
    if factors:
        for factor in factors: st.write(f"• {factor}")
    else:
        st.write("No implemented segmentation factors were detected.")

    transformer = pipeline.named_steps["preprocess"]
    transformed = transformer.transform(frame)
    names = transformer.get_feature_names_out()
    explanation = shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
    contributions = pd.DataFrame({"Feature": names, "SHAP value": explanation.values[0]})
    contributions["Absolute contribution"] = contributions["SHAP value"].abs()
    top = contributions.nlargest(8, "Absolute contribution").sort_values("SHAP value")
    st.subheader("Patient-specific model explanation")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["Feature"], top["SHAP value"], color=np.where(top["SHAP value"] >= 0, "#d95f5f", "#4c78a8"))
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Contribution to XGBoost model output (log-odds scale)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.download_button("Download patient report", make_pdf(patient, probability, classification, segment, factors),
                       file_name="stroke_prediction_prototype_report.pdf", mime="application/pdf")

with st.expander("Validated model-evaluation results"):
    selected = results.loc[results["Model"] == "XGBoost"].iloc[0]
    cols = st.columns(5)
    for col, label in zip(cols, ["ROC-AUC", "PR-AUC", "Recall (Sensitivity)", "Precision", "F1-Score"]):
        col.metric(label, f"{selected[label]:.3f}")
    st.dataframe(results, width="stretch", hide_index=True)
    st.image(str(OUTPUT / "final_xgboost_confusion_matrix.png"), caption="Independent test-set confusion matrix")
    st.image(str(OUTPUT / "roc_curve_comparison.png"), caption="Independent test-set ROC comparison")
    st.image(str(OUTPUT / "precision_recall_curve_comparison.png"), caption="Independent test-set precision-recall comparison")

with st.expander("Classification-threshold sensitivity analysis"):
    st.write(
        "This descriptive analysis shows how alternative thresholds change "
        "recall, precision, F1-score, false positives, and false negatives. "
        "The 0.50 threshold is retained because 0.30 and 0.40 provide the same "
        "82% recall with more false positives, while thresholds above 0.50 "
        "increase clinically important false negatives."
    )
    display_columns = ["Threshold", "Precision", "Recall (Sensitivity)", "F1-Score", "FP", "FN"]
    st.dataframe(
        thresholds[display_columns].style.format({
            "Threshold": "{:.2f}", "Precision": "{:.2%}",
            "Recall (Sensitivity)": "{:.2%}", "F1-Score": "{:.2%}",
        }),
        width="stretch",
        hide_index=True,
    )

with st.expander("XGBoost imbalance-strategy comparison"):
    st.write(
        "Five-fold stratified cross-validation on the training partition compared "
        "SMOTE alone, class weighting alone, and their combination. Class weighting "
        "alone was retained because it produced the strongest ROC-AUC, PR-AUC, "
        "F1-score, and balanced accuracy without double-correcting the minority class."
    )
    st.dataframe(imbalance, width="stretch", hide_index=True)

render_research_evidence(ROOT)
