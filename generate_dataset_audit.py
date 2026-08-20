from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/healthcare-dataset-stroke-data.csv")
OUT = Path("dataset_audit_outputs")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
df = pd.read_csv(DATA_PATH)

# Core audit tables requested for the methodology and implementation chapters.
df.head(5).to_csv(OUT / "dataset_head_5.csv", index=False)

schema = pd.DataFrame(
    {
        "Column": df.columns,
        "Original dtype": [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }
)
schema.to_csv(OUT / "dataset_schema.csv", index=False)

numeric_summary = df.select_dtypes(include="number").describe().T.reset_index()
numeric_summary = numeric_summary.rename(columns={"index": "Variable"})
numeric_summary.to_csv(OUT / "numerical_summary_statistics.csv", index=False)

missing = df.isna().sum().rename("Missing count").reset_index()
missing.columns = ["Variable", "Missing count"]
missing["Missing percent"] = missing["Missing count"] / len(df) * 100
missing.to_csv(OUT / "missing_values.csv", index=False)

frequency_rows = []
for column in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    counts = df[column].fillna("Missing").value_counts(dropna=False)
    for value, count in counts.items():
        frequency_rows.append(
            {
                "Variable": column,
                "Category": value,
                "Count": int(count),
                "Percent": float(count / len(df) * 100),
            }
        )
pd.DataFrame(frequency_rows).to_csv(OUT / "categorical_frequency_counts.csv", index=False)

preprocessing_audit = pd.DataFrame(
    [
        ["id", "int64", "Excluded identifier", "Not a predictor"],
        ["stroke", "int64", "Retained as binary target", "Never a predictor"],
        ["age", "float64", "Median imputation then StandardScaler", "Scaled numeric feature"],
        ["avg_glucose_level", "float64", "Median imputation then StandardScaler", "Scaled numeric feature"],
        ["bmi", "float64", "Training-median imputation then StandardScaler", "Scaled numeric feature"],
        ["hypertension", "int64", "Passed through unchanged as 0/1", "Binary model feature"],
        ["heart_disease", "int64", "Passed through unchanged as 0/1", "Binary model feature"],
        ["gender", "object", "Most-frequent imputation and one-hot encoding", "2 indicator features; Female reference"],
        ["ever_married", "object", "Most-frequent imputation and one-hot encoding", "1 indicator feature; No reference"],
        ["work_type", "object", "Most-frequent imputation and one-hot encoding", "4 indicators; Govt_job reference"],
        ["Residence_type", "object", "Most-frequent imputation and one-hot encoding", "1 indicator; Rural reference"],
        ["smoking_status", "object", "Most-frequent imputation and one-hot encoding", "3 indicators; Unknown reference"],
    ],
    columns=["Original variable", "Original dtype", "Transformation", "Model representation"],
)
preprocessing_audit.to_csv(OUT / "preprocessing_and_type_changes.csv", index=False)


def save_count(column, title, filename, order=None):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, order=order, color="#4C9ED9")
    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Number of patients")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()


def save_hist(column, title, filename, bins=30):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue="stroke", bins=bins, multiple="stack", palette=["#4C9ED9", "#E35D6A"])
    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()


save_count("stroke", "Stroke Outcome Distribution", "01_stroke_distribution.png", order=[0, 1])
save_hist("age", "Age Distribution by Stroke Outcome", "02_age_distribution.png", bins=30)
save_hist("bmi", "BMI Distribution by Stroke Outcome", "03_bmi_distribution.png", bins=30)
save_hist("avg_glucose_level", "Average Glucose Distribution by Stroke Outcome", "04_glucose_distribution.png", bins=30)
save_count("hypertension", "Hypertension Distribution", "06_hypertension_distribution.png", order=[0, 1])
save_count("heart_disease", "Heart-Disease Distribution", "07_heart_disease_distribution.png", order=[0, 1])
save_count("smoking_status", "Smoking-Status Distribution", "08_smoking_status_distribution.png")
save_count("work_type", "Work-Type Distribution", "09_work_type_distribution.png")
save_count("gender", "Gender Distribution", "10_gender_distribution.png")

# Corrected project-specific five-factor segmentation. It is descriptive only.
seg = df.copy()
seg["risk_score"] = (
    (seg["age"] >= 60).astype(int)
    + (seg["bmi"] >= 30).astype(int)
    + (seg["hypertension"] == 1).astype(int)
    + (seg["heart_disease"] == 1).astype(int)
    + (seg["avg_glucose_level"] >= 200).astype(int)
)
seg["patient_segment"] = pd.cut(
    seg["risk_score"], bins=[-1, 1, 2, 5], labels=["Low", "Medium", "High"]
)
segment_counts = seg["patient_segment"].value_counts().reindex(["Low", "Medium", "High"])
segment_counts.rename("Count").reset_index().to_csv(OUT / "patient_segment_counts.csv", index=False)
plt.figure(figsize=(8, 5))
sns.countplot(data=seg, x="patient_segment", order=["Low", "Medium", "High"], palette=["#63BE7B", "#FFD966", "#F8696B"], hue="patient_segment", legend=False)
plt.title("Project-Specific Five-Factor Patient Segmentation")
plt.xlabel("Descriptive segment (not clinically validated)")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig(OUT / "05_patient_segment_distribution.png", dpi=300)
plt.close()

numeric_for_corr = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_for_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Original Numeric Variables")
plt.tight_layout()
plt.savefig(OUT / "11_correlation_heatmap.png", dpi=300)
plt.close()

print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Stroke cases: {int(df['stroke'].sum()):,}")
print(f"Non-stroke cases: {int((df['stroke'] == 0).sum()):,}")
print(f"BMI missing values: {int(df['bmi'].isna().sum()):,}")
print("\nNumerical means:")
print(df.select_dtypes(include="number").mean().to_string())
print(f"\nOutputs written to {OUT.resolve()}")
