import streamlit as st
import openai
import pickle
import os
import numpy as np

# Load ML Models Safely
models = {}

try:
    with open('Models/diabetes_model.sav', 'rb') as file:
        models['diabetes'] = pickle.load(file)
    with open('Models/heart_disease_model.sav', 'rb') as file:
        models['heart_disease'] = pickle.load(file)
except FileNotFoundError:
    st.error("🚨 Error: Model files not found. Ensure the 'Models' folder contains trained models.")

# OpenAI API Key (Replace with your actual key)
openai.api_key = "your_openai_api_key"

st.title("🩺 AI Doctor Chatbot")

# Chatbot conversation
def chat_with_ai(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical chatbot that collects symptoms and predicts diseases."},
                {"role": "user", "content": user_input},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error in AI response: {e}"

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input field
user_input = st.text_input("👤 Patient:", key="user_input")

if st.button("Ask AI Doctor"):
    if user_input:
        bot_reply = chat_with_ai(user_input)
        st.session_state["messages"].append(f"👤 You: {user_input}")
        st.session_state["messages"].append(f"🤖 Doctor Bot: {bot_reply}")

# Display conversation history
st.subheader("🗨️ Chat History")
for msg in st.session_state["messages"]:
    st.write(msg)

# Disease Prediction Section
st.header("🩺 Disease Prediction")

# Collecting user inputs for prediction
st.subheader("Enter Your Health Information")

# User input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=5)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=35)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=25.3)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=45)

# Diagnosis Prediction
if st.button("Diagnose Me"):
    try:
        sample_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        diabetes_prediction = models['diabetes'].predict(sample_data)
        result = "Diabetic" if diabetes_prediction[0] == 1 else "Not Diabetic"
        st.success(f"🔍 Diagnosis: {result}")
    except Exception as e:
        st.error(f"🚨 Error in prediction: {e}")

st.markdown("⚠️ *This AI chatbot is not a substitute for a real doctor. Consult a professional for medical advice.*")

