# ============================================================================
# Predicting First Time Stroke Using Machine Learning and Patient Segmentation
# Machine Learning Model Development
# =============================================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# ==================================================
# 1. Load Dataset
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
# 2. Separate Features and Target
# ==================================================

X = df.drop("stroke", axis=1)

y = df["stroke"]


print("\n===== Feature Shape =====")
print(X.shape)


print("\n===== Target Distribution =====")
print(y.value_counts())


# ==================================================
# 3. Convert Categorical Columns
# ==================================================

X = pd.get_dummies(X, drop_first=True)


print("\n===== After Encoding =====")
print(X.shape)


# ==================================================
# 4. Train Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n===== Training Data =====")
print(X_train.shape)

print("\n===== Testing Data =====")
print(X_test.shape)


# ==================================================
# 5. Apply SMOTE to Balance Training Data
# ==================================================

from imblearn.over_sampling import SMOTE


print("\n===== Before SMOTE =====")
print(y_train.value_counts())


smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\n===== After SMOTE =====")
print(y_train_smote.value_counts())


print("\n===== Balanced Training Shape =====")
print(X_train_smote.shape)


# ==================================================
# 6. Train Machine Learning Models
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

}


results = []


for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")


    model.fit(
        X_train_smote,
        y_train_smote
    )


    y_pred = model.predict(
        X_test
    )


    y_prob = model.predict_proba(
        X_test
    )[:,1]


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)


    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])



# ==================================================
# 7. Model Comparison
# ==================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)


print("\n===== Model Comparison =====")
print(results_df)

# ==================================================
# 8. Confusion Matrix and ROC Curve Evaluation
# ==================================================

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# Store trained models

trained_models = {}


for name, model in models.items():

    # Train again and save model
    model.fit(
        X_train_smote,
        y_train_smote
    )

    trained_models[name] = model



# ==================================================
# Confusion Matrix
# ==================================================

for name, model in trained_models.items():

    y_pred = model.predict(
        X_test
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print("\n==============================")
    print(name)
    print("==============================")

    print(cm)


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )


    disp.plot()

    plt.title(
        name + " - Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        name.replace(" ", "_") + "_confusion_matrix.png"
    )


    plt.close()



# ==================================================
# ROC Curve Comparison
# ==================================================

plt.figure(figsize=(8,6))


for name, model in trained_models.items():

    y_probability = model.predict_proba(
        X_test
    )[:,1]


    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )


    plt.plot(
        fpr,
        tpr,
        label=name
    )



plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve Comparison"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "ROC_Curve_Comparison.png"
)


plt.close()


# ==================================================
# Extract TN, FP, FN, TP Values
# ==================================================

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    TN, FP, FN, TP = cm.ravel()

    print("\n==============================")
    print(name)
    print("==============================")

    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)
    print("False Negative (FN):", FN)
    print("True Positive (TP):", TP)



print("\n===== Evaluation Completed =====")
print("Confusion matrices and ROC curve saved")

# ==================================================
# SHAP Explainability - XGBoost
# ==================================================

import shap


print("\n===== SHAP Analysis =====")


# Train XGBoost model again

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=10,
    random_state=42,
    eval_metric="logloss"
)


xgb_model.fit(
    X_train_smote,
    y_train_smote
)


# Create SHAP Explainer

explainer = shap.TreeExplainer(
    xgb_model
)


# Calculate SHAP values

shap_values = explainer.shap_values(
    X_test
)


# ==================================================
# SHAP Summary Plot
# ==================================================

plt.figure(figsize=(10,8))


shap.summary_plot(
    shap_values,
    X_test,
    show=False
)


plt.title(
    "SHAP Feature Importance - XGBoost"
)


plt.tight_layout()


plt.savefig(
    "SHAP_summary_plot.png"
)


plt.close()


print("\n===== SHAP Completed Successfully =====")
print("Saved: SHAP_summary_plot.png")


# ==================================================
# Save Final XGBoost Model
# ==================================================

joblib.dump(
    xgb_model,
    "stroke_xgboost_model.pkl"
)

joblib.dump(
    X_train_smote.columns,
    "model_features.pkl"
)

print("XGBoost model saved successfully")