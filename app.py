import streamlit as st
import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 520.0, 32.0)
embarked = st.selectbox("Embarked from", ["S", "C", "Q"])

if st.button("Predict"):
    data = pd.DataFrame([[
        pclass,
        0 if sex == "male" else 1,
        age, sibsp, parch, fare,
        {"S": 0, "C": 1, "Q": 2}[embarked]
    ]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.success(f"Survived ✅ (probability {prob:.0%})")
    else:
        st.error(f"Did not survive ❌ (probability {prob:.0%})")
