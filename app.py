import streamlit as st
import numpy as np
import openai
import joblib

# Load model and label encoder
clf = joblib.load("models/svm_model.joblib")
le = joblib.load("models/label_encoder.joblib")

# Connect to LM Studio
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"

st.title("🐺 NeuralNet Watchdog - Anomaly Detector")

st.markdown("Enter packet features manually below (42 values total):")

# Input vector (simulate user input with sliders)
user_input = []
for i in range(42):
    user_input.append(st.number_input(f"Feature {i}", value=0.0))

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = clf.predict(input_array)[0]
    label = le.inverse_transform([prediction])[0]

    st.success(f"🚦 Prediction: **{label.upper()}**")

    # Ask Hermes-3 to explain
    with st.spinner("Talking to Hermes..."):
        prompt = f"This is a network packet with features: {user_input}. The ML model predicted '{label}'. Can you explain why this might be considered {label}?"
        response = openai.ChatCompletion.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You're a cybersecurity expert."},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response['choices'][0]['message']['content']
        st.markdown("### 🤖 Hermes-3 Explains:")
        st.write(explanation)
