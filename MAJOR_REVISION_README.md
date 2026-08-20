# Major-revision system

## Defensible outcome terminology

The dataset records whether `stroke` is 0 or 1 but contains no variable that
identifies a first-ever event, a recurrent event, or prior stroke history.
Accordingly, the revised project uses **Stroke Risk Prediction Using Machine
Learning and Patient Segmentation**.

## Corrected experimental design

- Raw predictors are split 80:20 with stratification before any learned preprocessing.
- BMI imputation, scaling, one-hot encoding, and imbalance handling are contained inside model pipelines.
- Five-fold stratified cross-validation on the training partition is used for comparison and selection.
- The independent test partition is evaluated once after model selection.
- Logistic Regression and Random Forest use balanced class weights.
- KNN uses SMOTE only inside training folds.
- Final XGBoost uses a training-derived `scale_pos_weight` (approximately 19.54) and does **not** use SMOTE.
- Derived segmentation fields are excluded from predictive modelling. The model has 16 transformed features.

## Feature scaling and preprocessing

The implementation uses a `ColumnTransformer` inside each model pipeline. Age,
average glucose, and BMI are median-imputed and standardized with
`StandardScaler`. Hypertension and heart disease pass through as binary 0/1
predictors. Gender, marital status, work type, residence type, and smoking
status are most-frequent imputed and one-hot encoded. Because the transformer is
inside the cross-validation pipeline, its medians, means, standard deviations,
and category mappings are learned only from the relevant training fold. This is
particularly important for KNN, whose Euclidean distances would otherwise be
dominated by variables with larger numerical ranges.

## Complete feature list and leakage exclusions

The original claim of 30 predictors arose from using the segmented dataset as
the modelling input. That design has been replaced. The revised pipeline starts
from the raw dataset and uses 10 eligible variables: age, average glucose, BMI,
hypertension, heart disease, gender, marital status, work type, residence type,
and smoking status. One-hot encoding produces 16 model features, documented in
`revised_outputs/feature_manifest.csv`. The patient identifier and stroke target
are excluded from predictors. Age group, BMI category, glucose-risk group,
hypertension group, heart-disease group, lifestyle group, risk score, and
patient segment are also excluded. The rule-based segmentation is calculated
separately for presentation only after model prediction and cannot leak into
model training.

## Project-specific patient segmentation

The descriptive segmentation assigns one point for each of five conditions:
age 60 years or older, BMI at least 30 kg/m², hypertension, heart disease, and
average glucose at least 200 mg/dL. Scores 0–1 are labelled Low, 2 Medium, and
3–5 High. Age 60 is a pragmatic older-adult boundary used by this project; BMI
30 follows the WHO adult obesity boundary; and glucose 200 mg/dL is used as a
high-glucose flag informed by established diabetes-testing thresholds. The
presence of hypertension and heart disease reflects recognized stroke-risk
conditions. Equal weights were chosen for transparency and simple descriptive
counting, not because the factors have equal clinical effects. The framework
has not been derived, calibrated, or validated as a clinical stroke-risk score.
It must not be used for diagnosis or treatment decisions and remains separate
from the XGBoost probability.


## Final independent-test results

| Metric | Result |
|---|---:|
| Accuracy | 77.69% |
| Precision | 15.77% |
| Recall/Sensitivity | 82.00% |
| Specificity | 77.47% |
| F1-score | 26.45% |
| Balanced accuracy | 79.73% |
| ROC-AUC | 84.85% |
| PR-AUC | 28.76% |
| Confusion matrix | TN 753, FP 219, FN 9, TP 41 |

These results prioritize sensitivity on a severely imbalanced dataset. They do not establish clinical validity.

## Classification-threshold analysis

Seven thresholds from 0.20 to 0.80 are reported in `revised_outputs/threshold_analysis.csv`.
The 0.50 threshold is retained: thresholds 0.30 and 0.40 produce the same 82% recall with more false positives, while thresholds above 0.50 increase false negatives. This analysis is descriptive and does not retrain the model.

## SMOTE versus class weighting

`revised_outputs/xgboost_imbalance_strategy_comparison.csv` reports a five-fold
training-only comparison of SMOTE alone, class weighting alone, and SMOTE plus
`scale_pos_weight=10`. Class weighting alone was retained because it achieved
the strongest mean ROC-AUC (0.8457), PR-AUC (0.2158), F1-score (0.2508), and
balanced accuracy (0.7742). The combined strategy increased recall but reduced
precision, F1-score, ROC-AUC, and PR-AUC, consistent with over-correction after
SMOTE had already balanced the training folds.

## Reproduce the results

```bash
python train_revised.py
streamlit run app_revised.py
```

The generated model, tables, and figures are stored in `revised_outputs/`.

## Deployment

The Streamlit Community Cloud deployment uses app_new.py as its entry point. This compatibility entry point executes the revised application implemented in app.py. The deployed system loads the revised XGBoost pipeline and corresponding validated outputs. Legacy saved models and fixed historical performance values are not used.
