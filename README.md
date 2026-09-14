# Stroke Prediction Using Machine Learning & Patient Risk Segmentation

An end-to-end machine learning project for **stroke prediction, patient risk segmentation, and explainable AI (SHAP)**, developed as part of my Master of Information Technology project.

The project demonstrates the complete machine learning workflow from data preprocessing and model evaluation to explainability and deployment through a Streamlit web application.

## Project Overview

Stroke prediction is a challenging machine learning problem because stroke-positive cases are significantly less common than non-stroke cases.

This project develops and evaluates multiple machine learning models to identify individuals with elevated stroke risk while addressing class imbalance and model interpretability.

The final system combines:

- Machine Learning Prediction
- Patient Risk Segmentation
- SHAP Explainable AI
- Interactive Streamlit Application
- Model Evaluation Dashboard

## Technologies Used

**Programming & Data**
- Python
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- XGBoost
- Imbalanced-learn

**Explainable AI**
- SHAP

**Visualization**
- Matplotlib
- Altair

**Application & Deployment**
- Streamlit
- GitHub
- Streamlit Community Cloud

## Dataset

The project uses the **Healthcare Stroke Dataset** available from Kaggle.

**Records:** 5,110  
**Stroke-positive cases:** 249

The model uses ten original predictors:

- Gender
- Age
- Hypertension
- Heart disease
- Ever married
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status

The identifier field is excluded from modelling.

After categorical encoding, the final machine-learning pipeline contains **16 encoded input features**.

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of the identifier field
- BMI median imputation
- Categorical one-hot encoding
- Numerical scaling where required
- Stratified train/test splitting
- Class imbalance evaluation

SMOTE was evaluated experimentally within training folds.

The final XGBoost implementation uses the original imbalanced training data with a training-derived `scale_pos_weight` rather than applying SMOTE to the final model.

## Machine Learning Models

Four machine learning algorithms were evaluated:

| Model | Imbalance Strategy |
|---|---|
| Logistic Regression | Class weighting |
| Random Forest | Class weighting |
| K-Nearest Neighbors | SMOTE within training folds |
| XGBoost | `scale_pos_weight` |

Model evaluation used **five-fold stratified cross-validation** on the training partition together with a separate held-out test set.

## Final Model — XGBoost

XGBoost was selected for the final application because of its ability to capture nonlinear relationships and provide strong sensitivity to the minority stroke class.

Example cross-validation performance:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8459 |
| PR-AUC | 0.2206 |
| Recall | 0.7791 |
| Precision | 0.1532 |
| F1 Score | 0.2560 |
| Balanced Accuracy | 0.7792 |

At a classification threshold of **0.50**, the held-out test evaluation achieved approximately:

**Recall: 82.0%**

**Precision: 15.77%**

The relatively low precision reflects the highly imbalanced nature of the dataset and the project's emphasis on identifying potential stroke-risk cases.

## Patient Risk Segmentation

In addition to machine-learning probability, the application provides a separate rule-based patient risk segmentation.

Five risk conditions are considered:

- Age ≥ 60 years
- BMI ≥ 30 kg/m²
- Hypertension
- Heart disease
- Average glucose ≥ 200 mg/dL

Risk groups:

| Score | Segment |
|---|---|
| 0–1 | Low Risk |
| 2 | Medium Risk |
| 3–5 | High Risk |

The segmentation framework is **project-specific and has not been clinically validated**. It is intended to provide an understandable educational summary alongside the machine-learning prediction.

## Explainable AI — SHAP

SHAP is integrated into the application to improve model transparency.

The system provides:

- Global feature importance
- Local patient-level explanations
- Feature contribution visualization

SHAP values describe how individual features influence the model prediction. They represent **model associations and should not be interpreted as causal medical relationships**.

## Application Workflow

User Input  
↓  
Input Validation  
↓  
Preprocessing Pipeline  
↓  
XGBoost Prediction  
↓  
Stroke Risk Probability  
↓  
Patient Risk Segmentation  
↓  
SHAP Explanation  
↓  
Results & Patient Report

The application also provides access to model evaluation information and allows generation of a patient report.

## Repository Structure

```text
stroke-prediction/
│
├── data/
├── dataset_audit_outputs/
├── revised_outputs/
│
├── app.py
├── app_new.py
├── train_revised.py
├── compare_xgboost_imbalance_strategies.py
├── dashboard_research_evidence.py
├── generate_dataset_audit.py
├── generate_model_comparison_figures.py
├── generate_revised_report_figures.py
├── requirements.txt
│
├── MAJOR_REVISION_README.md
└── EXAMINER_MAJOR_REVISION_COMPLIANCE.md


## Key Project Features
End-to-end machine learning workflow
Multiple model comparison
Stratified cross-validation
Class imbalance handling
Threshold-based evaluation
XGBoost classification
Patient risk segmentation
SHAP explainable AI
Interactive Streamlit interface
Model performance visualization
Automated patient report generation
Cloud deployment

## Project Limitations
The dataset contains only 249 stroke-positive cases, which limits the amount of minority-class information available for model training.

Additional limitations include:
Cross-sectional dataset
No distinction between first-time and recurrent stroke
Limited clinical variables
No medication or family-history information
No imaging or laboratory variables
Dataset representativeness is uncertain
No external clinical validation

Therefore, this application should not be used for medical diagnosis or clinical decision-making.


## Future Improvements
Future development could include:

Larger external datasets
External model validation
Prospective clinical data
Additional clinical predictors
Probability calibration
Hyperparameter optimization
Cost-sensitive learning
Ensemble imbalance strategies
Decision-curve analysis
Fairness assessment
Clinical validation of the segmentation framework


## Live Application
Streamlit Application:
https://stroke-risk-prediction-segmentation.streamlit.app/

##Skills Demonstrated
Python • Machine Learning • XGBoost • SHAP • Data Analysis • Model Evaluation • Data Visualization • Streamlit • GitHub • Explainable AI


## Disclaimer
This project was developed for academic, educational, and research purposes.
The machine-learning predictions and patient segmentation results have not been clinically validated and should not be interpreted as medical diagnoses or professional medical advice.

## Developed by Thanusyia Ramakrishnan
Master of Information Technology Project
