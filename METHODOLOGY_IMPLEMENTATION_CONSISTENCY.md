# Canonical workflow for Chapters 3, 5, and 6

All three chapters must describe the same implemented workflow.

| Stage | Canonical implementation | Prohibited superseded statement |
|---|---|---|
| Source data | Raw Kaggle dataset: 5,110 observations, 249 stroke-positive | Segmented dataset used for predictive training |
| Identifier/target | `id` excluded; `stroke` used only as target | ID or stroke included as a predictor |
| Holdout design | 80:20 stratified split, random state 42, before learned preprocessing | Imputation/encoding/scaling fitted before the split |
| BMI | Median imputer fitted within each training fold | One median calculated from the full dataset for modelling |
| Numeric scaling | Age, glucose, and BMI standardized with training-fold `StandardScaler` | No scaling, or scaling fitted on the complete dataset |
| Binary variables | Hypertension and heart disease passed through as 0/1 | Re-encoded segmentation labels used as predictors |
| Categorical encoding | Most-frequent imputation plus one-hot encoding inside the pipeline | `pd.get_dummies` applied to the segmented dataset before validation |
| Feature set | 10 original eligible predictors become 16 model features | 30 predictors including derived segmentation variables |
| Leakage exclusions | Risk score, patient segment, and all derived risk groups excluded | Patient segment or risk score used for model training |
| Cross-validation | Five-fold stratified CV on the 80% training partition | Model selected from the independent test results |
| Imbalance handling | LR/RF balanced weights; KNN training-fold SMOTE; final XGBoost training-derived class weight | Final XGBoost trained on 50:50 SMOTE data with weight 10 |
| XGBoost selection | Highest mean training-CV ROC-AUC and PR-AUC | Selected because it was assumed superior or because of test accuracy |
| Independent test | 1,022 untouched observations evaluated once after selection | Repeated test-set use for tuning or model selection |
| Threshold | Descriptive analysis 0.20–0.80; 0.50 retained | Threshold chosen solely to maximize accuracy |
| SHAP | TreeExplainer applied to the fitted XGBoost and transformed 16-feature data | SHAP interpreted as medical causation |
| Segmentation | Separate five-factor project-specific descriptive framework | Clinically validated stroke score or XGBoost input |
| Deployment | Streamlit loads the saved complete pipeline and accepts all 10 raw predictors | Manual construction of partial features with unspecified fields set to zero |
| Clinical status | Academic prototype; no external/prospective clinical validation | Diagnostic, screening-ready, or clinically validated system |
| Outcome terminology | Stroke risk/status prediction; dataset cannot distinguish first-ever from recurrent stroke | Unsupported “first-time” or “first-ever” stroke claims |

## Chapter roles

- Chapter 3 explains the planned leakage-controlled methods and selection criteria.
- Chapter 5 provides implementation evidence, cross-validation, independent-test results, threshold analysis, SHAP output, and deployment evidence.
- Chapter 6 interprets the same results, limitations, contributions, and future validation requirements without introducing different methods or numbers.
