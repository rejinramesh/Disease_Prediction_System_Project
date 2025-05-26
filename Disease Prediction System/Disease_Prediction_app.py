import streamlit as st
import pickle


# --- Set background image using a direct link ---
def set_background_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background_from_url(
    "https://i.ibb.co/5XQF0TsX/4dac5deae87b71af4498187b28842170.jpg"
)


# Optional: Apply additional CSS styling from file
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


local_css("style.css")

# --- Load models ---
heart_model = pickle.load(open("models/random_forest_Heart_Disease_model.pkl", "rb"))
diabetes_model = pickle.load(open("models/randomforestmodel_diabetes.pkl", "rb"))
cancer_model = pickle.load(
    open("models/logistic_regression_breast_cancer_model.pkl", "rb")
)

st.title("Disease Prediction System")

disease = st.selectbox(
    "Select a Disease to Predict:",
    ["-- Choose --", "Heart Disease", "Diabetes", "Breast Cancer"],
)


def to_float(values):
    return [float(v) for v in values]


# --- Heart Disease Form ---
if disease == "Heart Disease":
    st.subheader("Heart Disease Inputs (XGBoost Selected Features)")

    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    )
    ca = st.number_input(
        "Number of Major Vessels Colored by Fluoroscopy (ca) [0–3]",
        min_value=0,
        max_value=3,
    )
    thal = st.selectbox(
        "Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise (oldpeak) [0.0–6.0]",
        min_value=0.0,
        max_value=6.0,
        step=0.1,
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy (LVH)"],
    )
    age = st.number_input("Age (in years) [0–120]", min_value=0, max_value=120)
    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach) [60–220]", min_value=60, max_value=220
    )

    if st.button("Predict"):
        features = to_float(
            [
                [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic",
                ].index(cp),
                ca,
                ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
                + 3,  # thal encoded as 3, 6, 7 in common datasets
                ["Upsloping", "Flat", "Downsloping"].index(slope),
                1 if exang == "Yes" else 0,
                1 if sex == "Male" else 0,
                oldpeak,
                [
                    "Normal",
                    "ST-T Abnormality",
                    "Left Ventricular Hypertrophy (LVH)",
                ].index(restecg),
                age,
                thalach,
            ]
        )
        prediction = heart_model.predict([features])[0]
        st.success(
            "Positive for Heart Disease"
            if prediction == 1
            else "Negative for Heart Disease"
        )

# --- Diabetes Form ---
elif disease == "Diabetes":
    st.subheader("Diabetes Inputs")
    age = st.number_input("Age (in years) [0–120]", min_value=0, max_value=120)
    sex = st.selectbox("Sex (Male or Female)", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension (High BP)? (Yes/No)", ["Yes", "No"])
    heart_disease = st.selectbox("Has Heart Disease? (Yes/No)", ["Yes", "No"])
    smoking_history = st.selectbox(
        "Smoking History",
        ["Never", "Former", "Current", "Not Available", "Ever", "Unknown"],
    )
    bmi = st.number_input(
        "BMI (Body Mass Index) [10.0–50.0]", min_value=10.0, max_value=50.0, step=0.1
    )
    hba1c = st.number_input(
        "HbA1c Level (% avg. glucose 2–3 months) [4.0–15.0]",
        min_value=4.0,
        max_value=15.0,
        step=0.1,
    )
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL) [50–300]", min_value=50, max_value=300
    )

    if st.button("Predict"):
        features = to_float(
            [
                age,
                1 if sex == "Male" else 0,
                1 if hypertension == "Yes" else 0,
                1 if heart_disease == "Yes" else 0,
                [
                    "Never",
                    "Former",
                    "Current",
                    "Not Available",
                    "Ever",
                    "Unknown",
                ].index(smoking_history),
                bmi,
                hba1c,
                glucose,
            ]
        )
        prediction = diabetes_model.predict([features])[0]
        st.success(
            "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
        )

# --- Breast Cancer Form ---
elif disease == "Breast Cancer":
    st.subheader("Breast Cancer Inputs (Top 10 XGBoost Features)")

    radius3 = st.number_input(
        "radius3 (Worst Radius) [7.0–30.0]", min_value=7.0, max_value=30.0, step=0.1
    )
    perimeter3 = st.number_input(
        "perimeter3 (Worst Perimeter) [50.0–200.0]",
        min_value=50.0,
        max_value=200.0,
        step=0.1,
    )
    concave_points3 = st.number_input(
        "concave_points3 (Worst Concave Points) [0.01–0.30]",
        min_value=0.01,
        max_value=0.30,
        step=0.001,
    )
    concave_points1 = st.number_input(
        "concave_points1 (Mean Concave Points) [0.01–0.20]",
        min_value=0.01,
        max_value=0.20,
        step=0.001,
    )
    area3 = st.number_input(
        "area3 (Worst Area) [100–2500]", min_value=100.0, max_value=2500.0, step=1.0
    )
    texture3 = st.number_input(
        "texture3 (Worst Texture) [10.0–50.0]", min_value=10.0, max_value=50.0, step=0.1
    )
    concavity3 = st.number_input(
        "concavity3 (Worst Concavity) [0.01–0.60]",
        min_value=0.01,
        max_value=0.60,
        step=0.01,
    )
    concave_points2 = st.number_input(
        "concave_points2 (SE Concave Points) [0.001–0.15]",
        min_value=0.001,
        max_value=0.15,
        step=0.001,
    )
    radius2 = st.number_input(
        "radius2 (SE Radius) [0.1–3.5]", min_value=0.1, max_value=3.5, step=0.1
    )
    area1 = st.number_input(
        "area1 (Mean Area) [100–2000]", min_value=100.0, max_value=2000.0, step=1.0
    )

    if st.button("Predict"):
        features = to_float(
            [
                radius3,
                perimeter3,
                concave_points3,
                concave_points1,
                area3,
                texture3,
                concavity3,
                concave_points2,
                radius2,
                area1,
            ]
        )
        prediction = cancer_model.predict([features])[0]
        st.success(
            "Malignant (Breast Cancer)" if prediction == 1 else "Benign (No Cancer)"
        )

# --- Disclaimer ---
st.warning(
    "This prediction is based on statistical models and may not be 100% accurate. "
    "Please consult a medical professional for a proper diagnosis and treatment."
)
