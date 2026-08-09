# preprocessing.py

# ==================================================
# Stroke Prediction Using Machine Learning
# Data Preprocessing Pipeline
# ==================================================

# Import Libraries

import pandas as pd
import numpy as np


# ==================================================
# 1. Load Dataset
# ==================================================

file_path = "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(file_path)


print("===== Original Dataset Shape =====")
print(df.shape)


print("\n===== First 5 Records =====")
print(df.head())


# ==================================================
# 2. Understand Dataset Information
# ==================================================

print("\n===== Dataset Information =====")
print(df.info())


# ==================================================
# 3. Statistical Summary
# ==================================================

print("\n===== Statistical Summary =====")
print(df.describe())


# ==================================================
# 4. Check Missing Values
# ==================================================

print("\n===== Missing Values =====")
print(df.isnull().sum())


# ==================================================
# 5. Handle Missing BMI Values
# ==================================================

df["bmi"] = df["bmi"].fillna(df["bmi"].median())


print("\n===== Missing Values After BMI Cleaning =====")
print(df.isnull().sum())

# ==================================================
# 6. Handle Duplicate Records
# ==================================================

print("\n===== Duplicate Records =====")
print(df.duplicated().sum())

# ==================================================
# 7. Remove Patient ID
# ==================================================

df = df.drop("id", axis=1)

print("\n===== Final Columns =====")
print(df.columns)

# ==================================================
# 8. Encode Categorical Variables
# ==================================================

print("\n===== Encoding Categorical Variables =====")

df = pd.get_dummies(
    df,
    drop_first=True
)

print("\n===== Dataset After Encoding =====")
print(df.head())

print("\n===== Final Dataset Shape =====")
print(df.shape)


# ==================================================
# 9. Save Processed Dataset
# ==================================================

output_file = "processed_stroke_data.csv"

df.to_csv(output_file, index=False)

print("\n===== Preprocessing Completed =====")
print("Saved file:", output_file)