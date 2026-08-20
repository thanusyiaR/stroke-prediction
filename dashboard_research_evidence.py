"""Dataset-audit and full model-comparison sections for the revised dashboard."""
from pathlib import Path

import pandas as pd
import streamlit as st


def render_research_evidence(project_root):
    root = Path(project_root)
    audit = root / "dataset_audit_outputs"
    model_out = root / "revised_outputs"

    with st.expander("Dataset audit, descriptive statistics, and preprocessing evidence"):
        st.write(
            "All descriptive evidence below is generated from the original 5,110-row "
            "dataset. Predictive preprocessing is performed within training-only pipelines to prevent data leakage; patient segmentation is calculated separately and excluded from model training."
        )
        head = pd.read_csv(audit / "dataset_head_5.csv")
        schema = pd.read_csv(audit / "dataset_schema.csv")
        missing = pd.read_csv(audit / "missing_values.csv")
        numeric = pd.read_csv(audit / "numerical_summary_statistics.csv")
        changes = pd.read_csv(audit / "preprocessing_and_type_changes.csv")
        segments = pd.read_csv(audit / "patient_segment_counts.csv")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset rows", "5,110")
        c2.metric("Original columns", "12")
        c3.metric("Stroke cases", "249")
        c4.metric("Missing BMI", "201")

        st.markdown("**First five original records**")
        st.dataframe(head, width="stretch", hide_index=True)
        st.markdown("**Original schema and missingness**")
        st.dataframe(schema, width="stretch", hide_index=True)
        st.dataframe(missing, width="stretch", hide_index=True)
        st.markdown("**Numerical summary statistics, including means**")
        st.dataframe(numeric, width="stretch", hide_index=True)
        st.markdown("**Preprocessing and datatype/feature-representation changes**")
        st.dataframe(changes, width="stretch", hide_index=True)
        st.markdown("**Corrected five-factor descriptive segment counts**")
        st.dataframe(segments, width="stretch", hide_index=True)

    with st.expander("Exploratory data-analysis figures"):
        figures = [
            ("01_stroke_distribution.png", "Stroke outcome distribution"),
            ("02_age_distribution.png", "Age distribution by stroke outcome"),
            ("03_bmi_distribution.png", "BMI distribution by stroke outcome"),
            ("04_glucose_distribution.png", "Average glucose distribution by stroke outcome"),
            ("05_patient_segment_distribution.png", "Corrected five-factor descriptive segmentation"),
            ("06_hypertension_distribution.png", "Hypertension distribution"),
            ("07_heart_disease_distribution.png", "Heart-disease distribution"),
            ("08_smoking_status_distribution.png", "Smoking-status distribution"),
            ("09_work_type_distribution.png", "Work-type distribution"),
            ("10_gender_distribution.png", "Gender distribution"),
            ("11_correlation_heatmap.png", "Original numeric-variable correlation heatmap"),
        ]
        for start in range(0, len(figures), 2):
            columns = st.columns(2)
            for col, (filename, caption) in zip(columns, figures[start:start + 2]):
                col.image(str(audit / filename), caption=caption, width="stretch")
        st.caption(
            "The patient-segmentation chart is a project-specific descriptive summary. "
            "It is not a clinically validated stroke-risk score and is excluded from predictive training."
        )

    with st.expander("Complete four-model visual comparison"):
        table = pd.read_csv(model_out / "four_model_report_table.csv")
        st.dataframe(table, width="stretch", hide_index=True)
        st.image(
            str(model_out / "four_model_metric_comparison.png"),
            caption="Accuracy, precision, recall, F1, ROC-AUC, and PR-AUC on the independent test set",
            width="stretch",
        )
        st.image(
            str(model_out / "four_model_confusion_matrix_comparison.png"),
            caption="Logistic Regression, Random Forest, KNN, and XGBoost confusion matrices",
            width="stretch",
        )
        st.caption(
            "The final XGBoost model was selected using five-fold stratified training-only "
            "cross-validation, not independent-test accuracy alone."
        )
