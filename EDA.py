# ==================================================
# Stroke Prediction Using Machine Learning
# Exploratory Data Analysis (EDA)
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. Load Segmented Dataset
# ==================================================

file_path = "segmented_stroke_data.csv"

df = pd.read_csv(file_path)


print("===== Dataset Shape =====")
print(df.shape)


print("\n===== Dataset Columns =====")
print(df.columns)


print("\n===== First 5 Records =====")
print(df.head())


# ==================================================
# 2. Dataset Information
# ==================================================

print("\n===== Dataset Information =====")
df.info()



# ==================================================
# 3. Statistical Summary
# ==================================================

print("\n===== Statistical Summary =====")
print(df.describe())



# ==================================================
# 4. Missing Value Check
# ==================================================

print("\n===== Missing Values =====")
print(df.isnull().sum())



# ==================================================
# 5. Stroke Distribution
# ==================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="stroke",
    data=df
)

plt.title("Stroke Distribution")

plt.xlabel("Stroke (0 = No Stroke, 1 = Stroke)")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("01_stroke_distribution.png")

plt.close()



# ==================================================
# 6. Age Group Distribution
# ==================================================

plt.figure(figsize=(7,4))

sns.countplot(
    x="age_group",
    data=df
)

plt.title("Patient Age Group Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("02_age_group_distribution.png")

plt.close()



# ==================================================
# 7. BMI Category Distribution
# ==================================================

plt.figure(figsize=(7,4))

sns.countplot(
    x="bmi_category",
    data=df
)

plt.title("BMI Category Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("03_bmi_category_distribution.png")

plt.close()



# ==================================================
# 8. Glucose Risk Distribution
# ==================================================

plt.figure(figsize=(7,4))

sns.countplot(
    x="glucose_risk",
    data=df
)

plt.title("Glucose Risk Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("04_glucose_risk_distribution.png")

plt.close()



# ==================================================
# 9. Patient Risk Segment Distribution
# ==================================================

plt.figure(figsize=(7,4))

sns.countplot(
    x="patient_segment",
    data=df
)

plt.title("Patient Risk Segment Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("05_patient_segment_distribution.png")

plt.close()



# ==================================================
# 10. Hypertension Analysis
# ==================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="hypertension",
    data=df
)

plt.title("Hypertension Distribution")

plt.xlabel("0 = No Hypertension, 1 = Hypertension")

plt.tight_layout()

plt.savefig("06_hypertension_distribution.png")

plt.close()



# ==================================================
# 11. Heart Disease Analysis
# ==================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="heart_disease",
    data=df
)

plt.title("Heart Disease Distribution")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.tight_layout()

plt.savefig("07_heart_disease_distribution.png")

plt.close()


# ==================================================
# 12. Smoking Status Analysis
# ==================================================

smoking_columns = [
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes"
]

smoking_data = df[smoking_columns].sum()


plt.figure(figsize=(7,4))

sns.barplot(
    x=smoking_data.index,
    y=smoking_data.values
)

plt.title("Smoking Status Distribution")

plt.xticks(rotation=45)

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("08_smoking_status_distribution.png")

plt.close()



# ==================================================
# 13. Work Type Analysis
# ==================================================

work_columns = [
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children"
]


work_data = df[work_columns].sum()


plt.figure(figsize=(8,4))


sns.barplot(
    x=work_data.index,
    y=work_data.values
)


plt.title("Work Type Distribution")

plt.xticks(rotation=45)

plt.ylabel("Number of Patients")

plt.tight_layout()


plt.savefig(
    "09_work_type_distribution.png"
)


plt.close()



# ==================================================
# 14. Gender Analysis
# ==================================================

gender_columns = [
    "gender_Male",
    "gender_Other"
]


gender_data = df[gender_columns].sum()


plt.figure(figsize=(6,4))


sns.barplot(
    x=gender_data.index,
    y=gender_data.values
)


plt.title("Gender Distribution")

plt.xticks(rotation=45)

plt.ylabel("Number of Patients")


plt.tight_layout()


plt.savefig(
    "10_gender_distribution.png"
)


plt.close()

# ==================================================
# 15. Correlation Heatmap
# ==================================================

plt.figure(figsize=(14,10))


# Include numerical + encoded Boolean variables

numeric_df = df.select_dtypes(
    include=["int64", "float64", "bool"]
)


correlation = numeric_df.corr()


sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)


plt.title(
    "Feature Correlation Heatmap"
)


plt.tight_layout()


plt.savefig(
    "11_correlation_heatmap.png"
)


plt.close()



# ==================================================
# Completed
# ==================================================

print("\n===== EDA Completed Successfully =====")
print("All graphs saved in project folder")