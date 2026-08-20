"""Compare XGBoost imbalance strategies using training-only stratified CV."""
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_outputs"
df = pd.read_csv(ROOT / "data" / "healthcare-dataset-stroke-data.csv")
X = df.drop(columns=["id", "stroke"]); y = df["stroke"].astype(int)
X_train, _, y_train, _ = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
ratio = (y_train == 0).sum() / (y_train == 1).sum()

pre = ColumnTransformer([
    ("numeric", SkPipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())]), ["age", "avg_glucose_level", "bmi"]),
    ("binary", "passthrough", ["hypertension", "heart_disease"]),
    ("categorical", SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]),
     ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]),
], verbose_feature_names_out=False)

base = dict(n_estimators=500, learning_rate=.02, max_depth=3, min_child_weight=5,
            subsample=.8, colsample_bytree=.8, reg_lambda=5, reg_alpha=.1,
            random_state=42, eval_metric="logloss", n_jobs=-1)
strategies = {
    "SMOTE only": Pipeline([("preprocess", pre), ("smote", SMOTE(random_state=42)),
                             ("model", XGBClassifier(**base, scale_pos_weight=1))]),
    "Class weighting only": Pipeline([("preprocess", pre),
        ("model", XGBClassifier(**base, scale_pos_weight=ratio))]),
    "SMOTE + class weighting (10)": Pipeline([("preprocess", pre),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(**base, scale_pos_weight=10))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "Recall": "recall",
           "Precision": "precision", "F1-Score": "f1", "Balanced Accuracy": "balanced_accuracy"}
rows=[]
for name, model in strategies.items():
    scores=cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    row={"Strategy":name}
    for label in scoring:
        values=scores[f"test_{label}"]
        row[f"{label} Mean"]=values.mean(); row[f"{label} SD"]=values.std(ddof=1)
    rows.append(row)
result=pd.DataFrame(rows)
result.to_csv(OUT / "xgboost_imbalance_strategy_comparison.csv", index=False)
print(result.to_string(index=False))
