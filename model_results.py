import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")


# Remove ID column
if "id" in df.columns:
    df = df.drop("id", axis=1)


# Handle missing BMI
df["bmi"] = df["bmi"].fillna(df["bmi"].median())


# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)


# Separate features and target

X = df.drop("stroke", axis=1)

y = df["stroke"]


# Load model

model = joblib.load(
    "stroke_xgboost_model.pkl"
)


# Match model features

model_features = joblib.load(
    "model_features.pkl"
)


for col in model_features:

    if col not in X.columns:

        X[col] = 0


X = X[model_features]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Prediction

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]


# Results

print(
    "Accuracy:",
    accuracy_score(y_test,y_pred)
)


print(
    "Precision:",
    precision_score(y_test,y_pred)
)


print(
    "Recall:",
    recall_score(y_test,y_pred)
)


print(
    "F1 Score:",
    f1_score(y_test,y_pred)
)


print(
    "ROC AUC:",
    roc_auc_score(y_test,y_prob)
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(y_test,y_pred)
)