# Examiner major-revision compliance matrix

| No. | Examiner requirement | Corrective action and evidence | Result impact |
|---:|---|---|---|
| 1 | Defensible final-model selection | Four models compared using five-fold stratified CV on training partition. Weighted XGBoost selected for highest mean ROC-AUC (0.8457) and PR-AUC (0.2158). | Replaces unsupported original selection. |
| 2 | SMOTE and class weighting | XGBoost comparison: SMOTE only, class weighting only, and combination. Weighting only retained; final XGBoost does not use SMOTE. | Produces revised final model. |
| 3 | Feature scaling | Numeric imputation and StandardScaler inside training-fold ColumnTransformer; binary passthrough; categorical one-hot encoding. | Prevents leakage and supports KNN. |
| 4 | Feature list/leakage | Ten original eligible predictors become 16 encoded model features. Identifier, target, risk score, patient segment, and all segmentation-derived groups excluded. | Removes 30-feature segmented-data design. |
| 5 | “First-time stroke” claim | Dataset has no first-ever/recurrent-stroke indicator. Title and claims changed to “Stroke Risk Prediction Using Machine Learning and Patient Segmentation.” | Terminology only; metrics unchanged. |
| 6 | Segmentation justification | Restored five-factor framework: age ≥60, BMI ≥30, hypertension, heart disease, glucose ≥200; equal weights; 0–1 Low, 2 Medium, 3–5 High. Explicitly project-specific and non-clinically validated. | Segmentation display only. |
| 7 | Threshold analysis | Thresholds 0.20–0.80 evaluated for precision, recall, F1, FP, and FN. Threshold 0.50 retained to preserve 82% recall with fewer FP than 0.30/0.40. | No retraining; documents operating trade-off. |
| 8 | Model validation | 80:20 stratified holdout plus five-fold stratified CV confined to training partition. Independent test used once after model selection. | Stronger generalization evidence. |
| 9 | Methodology/implementation consistency | Canonical workflow matrix established for Chapters 3, 5, and 6 covering preprocessing through deployment. | Report-wide correction. |
| 10 | Formatting/numbering/references | Required final audit of TOC, lists, headings, captions, equations, citations, references, duplicate numbering, and page fields after all content changes. | Formatting only; final shipping gate. |

## Revised independent-test results

- Accuracy: 77.69%
- Precision: 15.77%
- Recall: 82.00%
- F1-score: 26.45%
- Balanced accuracy: 79.73%
- ROC-AUC: 84.85%
- PR-AUC: 28.76%
- Confusion matrix: TN 753, FP 219, FN 9, TP 41

These results describe an academic prototype and do not establish clinical validity.
