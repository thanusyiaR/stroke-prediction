# ==================================================
# Stroke Prediction System
# Streamlit Application - Clean Version
# ==================================================

import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from reportlab.pdfgen import canvas
from io import BytesIO

# ==================================================
# Page Setup
# ==================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠"
)


# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "stroke_xgboost_model.pkl"
    )

    features = joblib.load(
        "model_features.pkl"
    )

    return model, features


model, model_features = load_model()



# ==================================================
# Title
# ==================================================

st.title(
    "🧠 Stroke Prediction Using Machine Learning"
)

st.write(
    "XGBoost Model with Explainable AI (SHAP)"
)

st.divider()



# ==================================================
# User Role
# ==================================================

role = st.sidebar.radio(
    "Login As",
    [
        "Doctor",
        "Administrator"
    ]
)



# ==================================================
# Doctor Dashboard
# ==================================================

if role == "Doctor":


    st.header(
        "👨‍⚕️ Doctor Dashboard"
    )


    st.subheader(
        "Patient Information"
    )


    age = st.number_input(
        "Age",
        1,
        120,
        50
    )


    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )


    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )


    glucose = st.number_input(
        "Average Glucose Level",
        50.0
    )


    bmi = st.number_input(
        "BMI",
        25.0
    )


    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
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



    # ==================================================
    # Prediction Button
    # ==================================================

    if st.button(
        "🔍 Predict Stroke Risk"
    ):


        input_data = pd.DataFrame({

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



        # Add missing columns

        for col in model_features:

            if col not in input_data.columns:

                input_data[col]=0



        input_data = input_data[
            model_features
        ]



        prediction = model.predict(
            input_data
        )


        probability = model.predict_proba(
            input_data
        )[0][1]


        risk = probability * 100



        st.divider()


        # ==================================================
        # Result
        # ==================================================

        st.subheader(
            "Stroke Risk Result"
        )


        if risk >= 50:

            st.error(
                "🔴 High Stroke Risk"
            )


        elif risk >=20:

            st.warning(
                "🟡 Medium Stroke Risk"
            )


        else:

            st.success(
                "🟢 Low Stroke Risk"
            )


        st.metric(
            "Stroke Probability",
            f"{risk:.2f}%"
        )



        # ==================================================
        # Segmentation
        # ==================================================

        st.divider()


        st.subheader(
            "👤 Patient Segmentation"
        )


        if (
            age >=60
            or hypertension==1
            or heart_disease==1
            or glucose>150
        ):

            segment="High Risk Patient"


        elif (
            age>=40
            or bmi>=25
            or glucose>100
        ):

            segment="Medium Risk Patient"


        else:

            segment="Low Risk Patient"



        st.info(
            segment
        )



        # ==================================================
        # Risk Factors
        # ==================================================

        st.divider()


        st.subheader(
            "⚠️ Risk Factor Summary"
        )


        if age>=60:
            st.write(
                "⚠️ Older age"
            )


        if bmi>=25:
            st.write(
                "⚠️ Overweight BMI"
            )


        if hypertension==1:
            st.write(
                "⚠️ Hypertension"
            )


        if heart_disease==1:
            st.write(
                "⚠️ Heart disease"
            )


        if smoking!="never smoked":
            st.write(
                "⚠️ Smoking history"
            )



        # ==================================================
        # SHAP
        # ==================================================

        st.divider()


        st.subheader(
            "🤖 Explainable AI (SHAP)"
        )


        try:

            shap_img = Image.open(
                "SHAP_summary_plot.png"
            )


            st.image(
                shap_img,
                caption="SHAP Feature Importance"
            )


        except:

            st.warning(
                "SHAP image not found"
            )

#

# ==================================================
# Administrator
# ==================================================

else:

    st.header(
        "🖥 Administrator Dashboard"
    )


    st.write(
        """
        Administrator Functions:

        ✅ Model performance

        ✅ ROC Curve

        ✅ Confusion Matrix

        ✅ SHAP Analysis
        """
    )