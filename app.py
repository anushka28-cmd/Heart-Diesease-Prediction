%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("❤️ Heart Disease Prediction App")

# Load dataset
df = pd.read_csv("heart.csv")

st.subheader("Dataset")
st.dataframe(df)

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Remove missing values
df_encoded = df_encoded.dropna()

# Features & Target
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {acc:.2%}")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 20, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])

bp = st.number_input("Resting BP", 80, 250, 120)

chol = st.number_input("Cholesterol", 0, 700, 200)

fast = st.selectbox("Fasting Blood Sugar", [0, 1])

ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

hr = st.number_input("Maximum Heart Rate", 60, 220, 150)

angina = st.selectbox("Exercise Angina", ["N", "Y"])

oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fast,
        "RestingECG": ecg,
        "MaxHR": hr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }])

    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
