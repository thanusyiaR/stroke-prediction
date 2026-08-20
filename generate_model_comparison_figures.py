"""Create report-ready comparison figures from the validated Mac test results."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_outputs"
RESULTS = OUT / "independent_test_results.csv"
df = pd.read_csv(RESULTS)

required = {
    "Model", "Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score",
    "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP",
}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing required result columns: {sorted(missing)}")

sns.set_theme(style="whitegrid")
label_map = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "KNN": "knn",
    "XGBoost": "xgboost",
}


def safe_name(model):
    return label_map.get(model, re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_"))


def draw_cm(ax, row, title):
    matrix = np.array([[int(row["TN"]), int(row["FP"])],
                       [int(row["FN"]), int(row["TP"])]])
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["No stroke", "Stroke"], yticklabels=["No stroke", "Stroke"],
        linewidths=0.5,
    )
    ax.set_title(f"{title}\nIndependent Test Set, Threshold = 0.50")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


# One confusion-matrix file per model.
for _, row in df.iterrows():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    draw_cm(ax, row, row["Model"])
    fig.tight_layout()
    fig.savefig(OUT / f"{safe_name(row['Model'])}_confusion_matrix.png", dpi=300)
    plt.close(fig)

# A single panel helps the examiner compare minority-class errors directly.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (_, row) in zip(axes.flat, df.iterrows()):
    draw_cm(ax, row, row["Model"])
fig.suptitle("Four-Model Confusion-Matrix Comparison", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "four_model_confusion_matrix_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Full metric comparison; percentages remain in the scientifically valid 0-1 scale.
metric_columns = [
    "Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "PR-AUC"
]
long = df[["Model", *metric_columns]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
fig, ax = plt.subplots(figsize=(13, 7))
sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Independent Test-Set Performance of Four Machine-Learning Models")
ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "four_model_metric_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Exact table used to build the figures, rounded only for presentation.
report_table = df[[
    "Model", "Imbalance Strategy", "Accuracy", "Precision",
    "Recall (Sensitivity)", "Specificity", "F1-Score", "Balanced Accuracy",
    "MCC", "ROC-AUC", "PR-AUC", "Brier Score", "TN", "FP", "FN", "TP",
]].copy()
report_table.to_csv(OUT / "four_model_report_table.csv", index=False)

print("Created individual confusion matrices for:")
for model in df["Model"]:
    print(f"  - {model}")
print("Created four_model_confusion_matrix_comparison.png")
print("Created four_model_metric_comparison.png")
print("Created four_model_report_table.csv")
