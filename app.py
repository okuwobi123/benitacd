import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.pkl")

# Feature list
features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# App title
st.title("💓 Heart Disease Prediction App")
st.markdown("Provide the following details to check for heart disease risk.")

# Input form with sliders, drop-downs, and other interactive components
with st.form("user_input_form"):
    # Age as a slider for better user interaction
    Age = st.slider("Age", min_value=1, max_value=120, value=30)

    # Sex dropdown
    Sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    # Chest Pain Type (cp) dropdown
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])

    # Resting Blood Pressure (trestbps) slider
    trestbps = st.slider("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)

    # Cholesterol (chol) slider
    chol = st.slider("Cholesterol (chol)", min_value=100, max_value=600, value=250)

    # Fasting Blood Sugar (fbs) dropdown
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])

    # Resting ECG results (restecg) dropdown
    restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])

    # Max Heart Rate Achieved (thalach) slider
    thalach = st.slider("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)

    # Exercise Induced Angina (exang) dropdown
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])

    # ST Depression (oldpeak) slider for more precision
    oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

    # Slope of ST segment (slope) dropdown
    slope = st.selectbox("Slope of ST segment (slope)", options=[0, 1, 2])

    # Number of major vessels colored by fluoroscopy (ca) dropdown
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])

    # Thalassemia (thal) dropdown
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    # Submit button
    submit = st.form_submit_button("Predict")

# If user clicks the button
if submit:
    user_input = {
        'age': Age,
        'sex': Sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("🩺 Prediction Result:")
    st.success("Disease Detected" if prediction == 1 else "No Heart Disease Detected")
    st.info(f"Probability of No Disease: **{probability[0] * 100:.2f}%**")
    st.info(f"Probability of Disease: **{probability[1] * 100:.2f}%**")
