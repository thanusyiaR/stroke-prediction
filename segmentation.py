# ==================================================
# Stroke Prediction Using Machine Learning
# Patient Segmentation Module
# ==================================================

import pandas as pd
import numpy as np


# ==================================================
# 1. Load Processed Dataset
# ==================================================

file_path = "processed_stroke_data.csv"

df = pd.read_csv(file_path)


print("===== Original Processed Dataset Shape =====")
print(df.shape)


print("\n===== First 5 Records =====")
print(df.head())


# ==================================================
# 2. Age Segmentation
# ==================================================

def age_group(age):

    if age < 18:
        return "Young"

    elif age < 45:
        return "Adult"

    elif age < 60:
        return "Middle Age"

    else:
        return "Senior"


df["age_group"] = df["age"].apply(age_group)


print("\n===== Age Group Distribution =====")
print(df["age_group"].value_counts())


# ==================================================
# 3. BMI Segmentation
# ==================================================

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obese"


df["bmi_category"] = df["bmi"].apply(bmi_category)


print("\n===== BMI Category Distribution =====")
print(df["bmi_category"].value_counts())


# ==================================================
# 4. Glucose Risk Segmentation
# ==================================================

def glucose_category(value):

    if value < 100:
        return "Normal"

    elif value < 200:
        return "High"

    else:
        return "Very High"


df["glucose_risk"] = df["avg_glucose_level"].apply(glucose_category)


print("\n===== Glucose Risk Distribution =====")
print(df["glucose_risk"].value_counts())


# ==================================================
# 5. Hypertension Risk Segmentation
# ==================================================

def hypertension_risk(value):

    if value == 1:
        return "Hypertension"

    else:
        return "Normal"


df["hypertension_group"] = df["hypertension"].apply(
    hypertension_risk
)


print("\n===== Hypertension Group Distribution =====")
print(df["hypertension_group"].value_counts())


# ==================================================
# 6. Heart Disease Risk Segmentation
# ==================================================

def heart_risk(value):

    if value == 1:
        return "Heart Disease"

    else:
        return "No Heart Disease"


df["heart_disease_group"] = df["heart_disease"].apply(
    heart_risk
)


print("\n===== Heart Disease Group Distribution =====")
print(df["heart_disease_group"].value_counts())


# ==================================================
# 7. Lifestyle Risk Segmentation
# ==================================================

def lifestyle_risk(row):

    if row["smoking_status_smokes"] == True:
        return "High Lifestyle Risk"

    elif row["smoking_status_formerly smoked"] == True:
        return "Moderate Lifestyle Risk"

    else:
        return "Lower Lifestyle Risk"


df["lifestyle_group"] = df.apply(
    lifestyle_risk,
    axis=1
)


print("\n===== Lifestyle Group Distribution =====")
print(df["lifestyle_group"].value_counts())


# ==================================================
# 8. Overall Patient Risk Segment
# ==================================================

def patient_segment(row):

    risk_score = 0


    # Age Risk
    if row["age"] >= 60:
        risk_score += 1


    # BMI Risk
    if row["bmi"] >= 30:
        risk_score += 1


    # Hypertension Risk
    if row["hypertension"] == 1:
        risk_score += 1


    # Heart Disease Risk
    if row["heart_disease"] == 1:
        risk_score += 1


    # Glucose Risk
    if row["avg_glucose_level"] >= 200:
        risk_score += 1


    if risk_score >= 3:
        return "High Risk"

    elif risk_score == 2:
        return "Medium Risk"

    else:
        return "Low Risk"



df["patient_segment"] = df.apply(
    patient_segment,
    axis=1
)


print("\n===== Patient Segment Distribution =====")
print(df["patient_segment"].value_counts())


# ==================================================
# 9. Save Segmented Dataset
# ==================================================

output_file = "segmented_stroke_data.csv"


df.to_csv(
    output_file,
    index=False
)


print("\n===== Segmentation Completed =====")
print("Saved file:", output_file)


# ==================================================
# 10. Final Dataset Information
# ==================================================

print("\n===== Final Dataset Shape =====")
print(df.shape)


print("\n===== Final Columns =====")
print(df.columns)