"""Reproducible, leakage-controlled stroke-model training and evaluation.

All preprocessing is learned inside each training fold. The independent test
set is held out until the model/imbalance strategy has been selected by
five-fold stratified cross-validation on the training set.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="Found unknown categories")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "revised_outputs"
OUT.mkdir(exist_ok=True)


def preprocessor():
    numeric = ["age", "avg_glucose_level", "bmi"]
    binary = ["hypertension", "heart_disease"]
    categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    return ColumnTransformer([
        ("numeric", SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("binary", "passthrough", binary),
        ("categorical", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def metric_row(name, strategy, y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "Model": name, "Imbalance Strategy": strategy, "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, prediction),
        "Specificity": tn / (tn + fp), "F1-Score": f1_score(y_true, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y_true, prediction),
        "MCC": matthews_corrcoef(y_true, prediction),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
        "Brier Score": brier_score_loss(y_true, probability),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


df = pd.read_csv(DATA_FILE)
duplicates = int(df.duplicated().sum())
missing_bmi = int(df["bmi"].isna().sum())
X = df.drop(columns=["id", "stroke"])
y = df["stroke"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())

models = {
    "Logistic Regression": ("class_weight='balanced'", Pipeline([
        ("preprocess", preprocessor()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])),
    "Random Forest": ("class_weight='balanced'", Pipeline([
        ("preprocess", preprocessor()),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                          class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])),
    "KNN": ("SMOTE within training folds", Pipeline([
        ("preprocess", preprocessor()), ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    ])),
    "XGBoost": ("training-derived scale_pos_weight only", Pipeline([
        ("preprocess", preprocessor()),
        ("model", XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=3, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5, reg_alpha=0.1,
            scale_pos_weight=imbalance_ratio, random_state=RANDOM_STATE,
            eval_metric="logloss", n_jobs=-1,
        )),
    ])),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "Recall": "recall",
    "Precision": "precision", "F1-Score": "f1", "Balanced Accuracy": "balanced_accuracy",
}
cv_rows = []
holdout_rows = []
fitted = {}

roc_fig, roc_ax = plt.subplots(figsize=(8, 6))
pr_fig, pr_ax = plt.subplots(figsize=(8, 6))
for name, (strategy, pipeline) in models.items():
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    row = {"Model": name, "Imbalance Strategy": strategy}
    for label in scoring:
        values = scores[f"test_{label}"]
        row[f"{label} Mean"] = values.mean()
        row[f"{label} SD"] = values.std(ddof=1)
    cv_rows.append(row)

    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    probability = pipeline.predict_proba(X_test)[:, 1]
    holdout_rows.append(metric_row(name, strategy, y_test, probability))
    RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=roc_ax)
    PrecisionRecallDisplay.from_predictions(y_test, probability, name=name, ax=pr_ax)

roc_ax.plot([0, 1], [0, 1], "k--", alpha=.5)
roc_ax.set_title("ROC Curves on the Independent Test Set")
roc_fig.tight_layout(); roc_fig.savefig(OUT / "roc_curve_comparison.png", dpi=200); plt.close(roc_fig)
pr_ax.axhline(y_test.mean(), color="k", linestyle="--", alpha=.5, label="Prevalence baseline")
pr_ax.set_title("Precision-Recall Curves on the Independent Test Set")
pr_fig.tight_layout(); pr_fig.savefig(OUT / "precision_recall_curve_comparison.png", dpi=200); plt.close(pr_fig)

cv_df = pd.DataFrame(cv_rows).sort_values(["ROC-AUC Mean", "PR-AUC Mean"], ascending=False)
holdout_df = pd.DataFrame(holdout_rows)
cv_df.to_csv(OUT / "cross_validation_results.csv", index=False)
holdout_df.to_csv(OUT / "independent_test_results.csv", index=False)

# Descriptive threshold-sensitivity analysis on the independent test set.
# This does not retrain or select the model; it documents the operational
# trade-off after the model was selected using training-only cross-validation.
threshold_rows = []
selected_probability = fitted["XGBoost"].predict_proba(X_test)[:, 1]
for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    row = metric_row(
        "XGBoost", "training-derived scale_pos_weight only",
        y_test, selected_probability, threshold=threshold,
    )
    threshold_rows.append(row)
pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_analysis.csv", index=False)

# XGBoost is selected strictly from training-set cross-validation results.
final_model = fitted["XGBoost"]
final_probability = final_model.predict_proba(X_test)[:, 1]
final_prediction = (final_probability >= 0.5).astype(int)
cm = confusion_matrix(y_test, final_prediction)

fig, ax = plt.subplots(figsize=(5.8, 5))
ax.imshow(cm, cmap="Blues")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center", fontsize=15)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No stroke", "Stroke"],
       yticklabels=["No stroke", "Stroke"], xlabel="Predicted", ylabel="Actual",
       title="Final XGBoost Confusion Matrix (Threshold = 0.50)")
fig.tight_layout(); fig.savefig(OUT / "final_xgboost_confusion_matrix.png", dpi=200); plt.close(fig)

prob_true, prob_pred = calibration_curve(y_test, final_probability, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.plot(prob_pred, prob_true, marker="o", label="XGBoost")
ax.set(xlabel="Mean predicted probability", ylabel="Observed stroke proportion",
       title="XGBoost Calibration on Independent Test Set")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "final_xgboost_calibration.png", dpi=200); plt.close(fig)

# Explain transformed test observations while retaining human-readable feature names.
transformer = final_model.named_steps["preprocess"]
x_test_transformed = transformer.transform(X_test)
feature_names = transformer.get_feature_names_out()
explainer = shap.TreeExplainer(final_model.named_steps["model"])
shap_values = explainer(x_test_transformed)
shap.summary_plot(shap_values, x_test_transformed, feature_names=feature_names, show=False)
plt.title("Final XGBoost SHAP Summary")
plt.tight_layout(); plt.savefig(OUT / "final_xgboost_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()

joblib.dump(final_model, OUT / "stroke_xgboost_pipeline.joblib")
metadata = {
    "dataset_records": len(df), "non_stroke_records": int((y == 0).sum()),
    "stroke_records": int((y == 1).sum()), "missing_bmi_before_imputation": missing_bmi,
    "duplicate_rows": duplicates, "training_records": len(X_train), "test_records": len(X_test),
    "training_stroke_records": int(y_train.sum()), "test_stroke_records": int(y_test.sum()),
    "random_state": RANDOM_STATE, "cv_folds": 5, "classification_threshold": 0.5,
    "scale_pos_weight": imbalance_ratio, "selected_model": "XGBoost",
    "selection_basis": "highest mean ROC-AUC and PR-AUC in five-fold stratified training-only cross-validation",
    "raw_input_columns": list(X.columns),
}
(OUT / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

print("\nCross-validation results (training set only):")
print(cv_df.to_string(index=False))
print("\nIndependent test results (reported once after selection):")
print(holdout_df.to_string(index=False))
print("\nDescriptive XGBoost threshold analysis:")
print(pd.DataFrame(threshold_rows).to_string(index=False))
print(f"\nOutputs written to {OUT}")
