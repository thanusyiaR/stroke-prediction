# =====================================================
# STROKE PREDICTION DASHBOARD
# XGBoost + SHAP Explainable AI
# =====================================================


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns


from io import BytesIO


from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)



# =====================================================
# PAGE CONFIGURATION
# =====================================================


st.set_page_config(

    page_title="Predicting First Time Stroke Using Machine Learning with Patient Segmentation",

    page_icon="🧠",

    layout="wide"

)



# =====================================================
# LOAD TRAINED MODEL
# =====================================================


@st.cache_resource
def load_model():

    model = joblib.load(
        "stroke_xgboost_model.pkl"
    )


    model_features = joblib.load(
        "model_features.pkl"
    )


    return model, model_features



model, model_features = load_model()



# =====================================================
# CREATE PDF REPORT FUNCTION
# =====================================================


def create_pdf(patient_data):


    buffer = BytesIO()


    doc = SimpleDocTemplate(
        buffer
    )


    styles = getSampleStyleSheet()


    story = []


    story.append(

        Paragraph(

            "Stroke Prediction Patient Report",

            styles["Title"]

        )

    )


    story.append(
        Spacer(1,20)
    )



    table_data = [

        [
            "Information",
            "Value"
        ]

    ]



    for key,value in patient_data.items():

        table_data.append(

            [

                str(key),

                str(value)

            ]

        )



    table = Table(
        table_data
    )



    table.setStyle(

        TableStyle(

            [

                (

                    "GRID",

                    (0,0),

                    (-1,-1),

                    1,

                    colors.black

                )

            ]

        )

    )



    story.append(
        table
    )



    doc.build(
        story
    )



    buffer.seek(0)



    return buffer



# =====================================================
# TITLE
# =====================================================


st.title(
    "🧠 Predicting First Time Stroke Using Machine Learning and Patient Segmentation"
)


st.caption(
    "Doctor Evaluation Support System:"
)

st.divider()



# =====================================================
# ABOUT MODEL
# =====================================================


with st.expander(
    "ℹ️ About the Stroke Prediction Model"
):


    st.write(

        """

        **System:**  
        Stroke Prediction Using Machine Learning and Patient Segmentation


        **Algorithm:**  
        XGBoost Classifier


        **Dataset:**  
        Healthcare Stroke Dataset


        **Records:**  
        5,110 patient records


        **Explainability:**  
        SHAP (SHapley Additive exPlanations)


        **Purpose:**  
        Predict first-time stroke probability and provide explainable AI insights.

        """

    )


# =====================================================
# DOCTOR DASHBOARD
# =====================================================


st.header(
    "👨‍⚕️ Doctor Dashboard"
)


st.subheader(
    "Patient Information"
)



left_col, right_col = st.columns(2)



with left_col:


    age = st.number_input(

        "Age",

        min_value=1,

        max_value=120,

        value=50

    )



    gender = st.selectbox(

        "Gender",

        [

            "Female",

            "Male"

        ]

    )



    hypertension = st.selectbox(

        "Hypertension",

        [0,1],

        format_func=lambda x:

        "Yes" if x == 1 else "No"

    )



    heart_disease = st.selectbox(

        "Heart Disease",

        [0,1],

        format_func=lambda x:

        "Yes" if x == 1 else "No"

    )


with right_col:


    glucose = st.number_input(

        "Average Glucose Level",

        min_value=50.0,

        max_value=350.0,

        value=100.0

    )



    bmi = st.number_input(

        "BMI",

        min_value=10.0,

        max_value=60.0,

        value=25.0

    )



    smoking = st.selectbox(

        "Smoking Status",

        [

            "never smoked",

            "formerly smoked",

            "smokes"

        ]

    )



    work = st.selectbox(

        "Work Type",

        [

            "Private",

            "Self-employed",

            "children",

            "Never_worked"

        ]

    )





st.divider()



predict_button = st.button(

    "🔍 Predict Stroke Risk",

    width="stretch"

)


# =====================================================
# START PREDICTION
# =====================================================


if predict_button:



    # =================================================
    # CREATE INPUT DATAFRAME
    # =================================================


    input_df = pd.DataFrame({


        "age":[age],


        "hypertension":[hypertension],


        "heart_disease":[heart_disease],


        "avg_glucose_level":[glucose],


        "bmi":[bmi],



        "gender_Male":[

            1 if gender=="Male" else 0

        ],



        "smoking_status_smokes":[

            1 if smoking=="smokes" else 0

        ],



        "smoking_status_formerly smoked":[

            1 if smoking=="formerly smoked" else 0

        ],



        "work_type_Private":[

            1 if work=="Private" else 0

        ],



        "work_type_Self-employed":[

            1 if work=="Self-employed" else 0

        ],



        "work_type_children":[

            1 if work=="children" else 0

        ],



        "work_type_Never_worked":[

            1 if work=="Never_worked" else 0

        ]

    })


    # Add missing training features


    for feature in model_features:


        if feature not in input_df.columns:

            input_df[feature]=0




    # Same order as training


    input_df = input_df[model_features]



    # =================================================
    # XGBOOST PREDICTION
    # =================================================


    probability = model.predict_proba(

        input_df

    )[0][1]



    risk_percentage = probability * 100



    # =================================================
    # CLINICAL RISK SUMMARY
    # =================================================


    st.divider()


    st.subheader(

        "⚠️ Clinical Risk Factor Summary"

    )



    risk_factors=[]



    if age >= 60:

        risk_factors.append(

            "Older Age"

        )


    if hypertension == 1:

        risk_factors.append(

            "Hypertension"

        )


    if heart_disease == 1:

        risk_factors.append(

            "Heart Disease"

        )


    if glucose > 150:

        risk_factors.append(

            "High Glucose Level"

        )


    if bmi >=25:

        risk_factors.append(

            "High BMI"

        )


    if smoking != "never smoked":

        risk_factors.append(

            "Smoking History"

        )



    clinical_score=len(risk_factors)




    if risk_factors:


        for factor in risk_factors:

            st.write(

                "⚠️ " + factor

            )


    else:


        st.success(

            "No major clinical risk factors detected"

        )



    # =================================================
    # MACHINE LEARNING RESULT
    # =================================================


    st.divider()


    st.subheader(

        "📈 Machine Learning Result"

    )



    if risk_percentage >=50:


        st.error(

            "🔴 High Stroke Probability"

        )


    elif risk_percentage >=20:


        st.warning(

            "🟡 Medium Stroke Probability"

        )


    else:


        st.success(

            "🟢 Low Stroke Probability"

        )



    st.metric(

        "Stroke Probability",

        f"{risk_percentage:.2f}%"

    )





    # =================================================
    # PATIENT SEGMENTATION
    # =================================================


    st.divider()


    st.subheader(

        "👤 Patient Segmentation"

    )



    if clinical_score >=4:


        patient_segment="🔴 High Risk Patient"



    elif clinical_score >=2:


        patient_segment="🟡 Medium Risk Patient"



    else:


        patient_segment="🟢 Low Risk Patient"




    st.info(

        patient_segment

    )


    # =================================================
    # SHAP EXPLAINABLE AI
    # =================================================


    st.divider()


    st.subheader(

        "🔍 SHAP Explainable AI"

    )



    try:



        explainer = shap.TreeExplainer(

            model

        )



        shap_values = explainer.shap_values(

            input_df

        )



        # XGBoost compatibility

        if isinstance(shap_values,list):

            shap_values = shap_values[1]




        shap_df = pd.DataFrame({


            "Risk Factor":

            input_df.columns,


            "SHAP Contribution":

            shap_values[0]


        })




        shap_df["Importance"] = (

            shap_df["SHAP Contribution"]

            .abs()

        )




        shap_df = shap_df.sort_values(

            by="Importance",

            ascending=False

        )




        top_features = shap_df.head(5)




        st.write(

            "Top factors influencing stroke prediction:"

        )



        st.dataframe(

            top_features[

                [

                    "Risk Factor",

                    "SHAP Contribution"

                ]

            ],

            width="stretch"

        )




        # SHAP Bar Chart


        fig,ax = plt.subplots(

            figsize=(8,4)

        )



        ax.barh(

            top_features["Risk Factor"],

            top_features["SHAP Contribution"]

        )



        ax.set_xlabel(

            "SHAP Contribution"

        )


        ax.set_ylabel(

            "Risk Factor"

        )


        ax.set_title(

            "Top Stroke Risk Contributors"

        )



        st.pyplot(

            fig

        )



    except Exception as e:


        st.warning(

            f"SHAP explanation unavailable: {e}"

        )


# =====================================================
# PDF REPORT
# =====================================================


if predict_button:


    st.divider()


    st.subheader(
        "📄 Patient Report"
    )



    patient_report = {


        "Age": age,


        "Gender": gender,


        "Hypertension":
        "Yes" if hypertension == 1 else "No",


        "Heart Disease":
        "Yes" if heart_disease == 1 else "No",


        "Glucose Level":
        glucose,


        "BMI":
        bmi,


        "Smoking Status":
        smoking,


        "Work Type":
        work,


        "Stroke Probability":
        f"{risk_percentage:.2f}%",


        "Patient Segment":
        patient_segment

    }



    pdf_file = create_pdf(

        patient_report

    )



    st.download_button(

        label="⬇️ Download Patient PDF Report",

        data=pdf_file,

        file_name="stroke_patient_report.pdf",

        mime="application/pdf",

        width="stretch"

    )



# =====================================================
# MODEL PERFORMANCE DASHBOARD
# =====================================================

st.divider()

st.header(
    "📊 Model Performance Dashboard"
)


try:

    # Load original dataset

    df = pd.read_csv(
        "healthcare-dataset-stroke-data.csv"
    )


    # Remove ID

    if "id" in df.columns:

        df = df.drop(
            "id",
            axis=1
        )


    # Fill missing BMI

    df["bmi"] = df["bmi"].fillna(
        df["bmi"].median()
    )


    # Encode categorical data

    df = pd.get_dummies(
        df,
        drop_first=True
    )


    X = df.drop(
        "stroke",
        axis=1
    )


    y = df["stroke"]



    # Match training features

    for feature in model_features:

        if feature not in X.columns:

            X[feature] = 0



    X = X[model_features]



    # Test split

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    # Prediction

    y_pred = model.predict(
        X_test
    )


    y_prob = model.predict_proba(
        X_test
    )[:,1]



    # Metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )



    # Display Metrics

    col1,col2,col3,col4,col5 = st.columns(5)


    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )


    col2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )


    col3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )


    col4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )


    col5.metric(
        "ROC-AUC",
        f"{roc_auc*100:.2f}%"
    )



    # =====================================================
    # CONFUSION MATRIX
    # =====================================================


    st.divider()

    st.subheader(
        "📋 Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    cm_df = pd.DataFrame(

        cm,

        index=[
            "Actual No Stroke",
            "Actual Stroke"
        ],

        columns=[
            "Predicted No Stroke",
            "Predicted Stroke"
        ]

    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )



    # Heatmap

    fig,ax = plt.subplots(
        figsize=(5,4)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        ax=ax

    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


    ax.set_title(
        "Confusion Matrix Heatmap"
    )


    st.pyplot(
        fig
    )



    # =====================================================
    # ROC CURVE
    # =====================================================


    st.divider()


    st.subheader(
        "📈 ROC Curve"
    )


    fpr,tpr,_ = roc_curve(

        y_test,

        y_prob

    )


    roc_value = auc(

        fpr,

        tpr

    )


    fig,ax = plt.subplots()


    ax.plot(

        fpr,

        tpr,

        label=f"AUC = {roc_value:.2f}"

    )


    ax.plot(

        [0,1],

        [0,1],

        linestyle="--"

    )


    ax.set_xlabel(
        "False Positive Rate"
    )


    ax.set_ylabel(
        "True Positive Rate"
    )


    ax.set_title(
        "Receiver Operating Characteristic Curve"
    )


    ax.legend()


    st.pyplot(
        fig
    )



except Exception as e:


    st.warning(
        f"Model Performance Dashboard unavailable: {e}"
    )

