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
    st.subheader("Heart Disease Inputs")
    age = st.number_input("Age (in years) [0–120]", min_value=0, max_value=120)
    sex = st.selectbox("Sex (Male or Female)", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type: Typical, Atypical, Non-anginal, Asymptomatic",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg) [90–200]", min_value=90, max_value=200
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dL) [100–600]", min_value=100, max_value=600
    )
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL? (Yes/No)", ["Yes", "No"])
    restecg = st.selectbox(
        "Resting ECG Results: Normal, ST-T Abnormality, LVH",
        ["Normal", "ST-T Abnormality", "LVH"],
    )
    thalach = st.number_input(
        "Max Heart Rate Achieved (bpm) [60–220]", min_value=60, max_value=220
    )
    exang = st.selectbox("Exercise-Induced Angina (Yes/No)", ["Yes", "No"])
    oldpeak = st.number_input(
        "ST Depression Relative to Rest [0.0–6.0]",
        min_value=0.0,
        max_value=6.0,
        step=0.1,
    )
    slope = st.selectbox(
        "Slope of ST Segment: Upsloping, Flat, Downsloping",
        ["Upsloping", "Flat", "Downsloping"],
    )
    ca = st.number_input(
        "Major Vessels Colored by Fluoroscopy [0–3]", min_value=0, max_value=3
    )
    thal = st.selectbox(
        "Thalassemia Type: Normal, Fixed Defect, Reversible Defect",
        ["Normal", "Fixed Defect", "Reversible Defect"],
    )

    if st.button("Predict"):
        features = to_float(
            [
                age,
                1 if sex == "Male" else 0,
                [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic",
                ].index(cp),
                trestbps,
                chol,
                1 if fbs == "Yes" else 0,
                ["Normal", "ST-T Abnormality", "LVH"].index(restecg),
                thalach,
                1 if exang == "Yes" else 0,
                oldpeak,
                ["Upsloping", "Flat", "Downsloping"].index(slope),
                ca,
                ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 3,
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
    st.subheader("Breast Cancer Inputs")
    radius = st.number_input(
        "Radius Mean (Size of Tumor) [6.0–30.0]",
        min_value=6.0,
        max_value=30.0,
        step=0.1,
    )
    texture = st.number_input(
        "Texture Mean (Gray Intensity Variation) [10.0–40.0]",
        min_value=10.0,
        max_value=40.0,
        step=0.1,
    )
    perimeter = st.number_input(
        "Perimeter Mean (Tumor Boundary) [50.0–200.0]",
        min_value=50.0,
        max_value=200.0,
        step=0.1,
    )
    area = st.number_input(
        "Area Mean (Tumor Surface Area) [100–2500]",
        min_value=100.0,
        max_value=2500.0,
        step=1.0,
    )
    compactness = st.number_input(
        "Compactness Mean (Perimeter² / Area) [0.01–0.35]",
        min_value=0.01,
        max_value=0.35,
        step=0.01,
    )
    concavity = st.number_input(
        "Concavity Mean (Edge Inward Curves) [0.01–0.50]",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
    )
    concave_points = st.number_input(
        "Concave Points Mean (Edge Irregularities) [0.001–0.20]",
        min_value=0.001,
        max_value=0.20,
        step=0.001,
    )
    symmetry = st.number_input(
        "Symmetry Mean (Tumor Symmetry) [0.1–0.5]",
        min_value=0.1,
        max_value=0.5,
        step=0.01,
    )
    fractal_dim = st.number_input(
        "Fractal Dimension (Boundary Complexity) [0.04–0.15]",
        min_value=0.04,
        max_value=0.15,
        step=0.01,
    )
    radius_se = st.number_input(
        "Radius SE (Radius Error) [0.1–3.0]", min_value=0.1, max_value=3.0, step=0.1
    )

    if st.button("Predict"):
        features = to_float(
            [
                radius,
                texture,
                perimeter,
                area,
                compactness,
                concavity,
                concave_points,
                symmetry,
                fractal_dim,
                radius_se,
            ]
        )
        prediction = cancer_model.predict([features])[0]
        st.success(
            "Malignant (Breast Cancer)" if prediction == "M" else "Benign (No Cancer)"
        )

# --- Disclaimer ---
st.warning(
    "This prediction is based on statistical models and may not be 100% accurate. "
    "Please consult a medical professional for a proper diagnosis and treatment."
)
