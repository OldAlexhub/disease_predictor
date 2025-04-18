import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import re
from datetime import datetime
import os

# --- Load assets ---
rfModel = joblib.load('rf_model.pkl')
data = pd.read_csv('training_data.csv')
data_symptoms = pd.read_csv('Diseases_Symptoms.csv')
all_symptoms = data.drop(columns=['prognosis', 'Unnamed: 133'], errors='ignore').columns.tolist()

# Prepare X_train and y_train for training accuracy and drift calculation
X_train = data.drop(columns=['prognosis', 'Unnamed: 133'], errors='ignore')
y_train = data['prognosis']
training_avg_symptoms = X_train.sum(axis=1).mean()

# --- Page Config ---
st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="wide")

# --- Style ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #00c8b4;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00b2a1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Monitoring Functions ---
def log_prediction(selected_symptoms, prediction, confidence):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms_selected": len(selected_symptoms),
        "prediction": prediction,
        "confidence": confidence
    }
    log_df = pd.DataFrame([log_entry])
    if not os.path.isfile('monitoring_log.csv'):
        log_df.to_csv('monitoring_log.csv', index=False)
    else:
        log_df.to_csv('monitoring_log.csv', mode='a', header=False, index=False)

def simple_drift_check(selected_symptoms, training_avg_symptoms):
    current_symptoms_count = len(selected_symptoms)
    if abs(current_symptoms_count - training_avg_symptoms) > (0.5 * training_avg_symptoms):
        st.warning("⚠️ Symptom selection pattern looks unusual compared to training data. Possible data drift.")

# --- Helper Functions ---
def create_input_vector(selected_symptoms, all_symptoms):
    return [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

def clean_disease_name(name):
    name = re.sub(r'\(.*?\)', '', name)  # remove anything in parentheses
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # remove punctuation
    name = name.strip().replace(" ", "+")
    return name.lower()

# --- Header ---
st.markdown("# 🧠 AI-Powered Disease Predictor")
st.markdown("Use intelligent symptom analysis to identify likely diseases and get expert-based treatment recommendations.")

# --- Symptom Input ---
st.markdown("### 👉 Select your current symptoms:")
selected_symptoms = []
cols = st.columns(4)
for i, symptom in enumerate(all_symptoms):
    label = symptom.replace('_', ' ').capitalize()
    if cols[i % 4].checkbox(label):
        selected_symptoms.append(symptom)

# --- Predict Button ---
if st.button("🔍 Analyze Symptoms & Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom to continue.")
    else:
        user_input = create_input_vector(selected_symptoms, all_symptoms)
        
        # Predict
        prediction = rfModel.predict([user_input])[0]
        prediction_proba = rfModel.predict_proba([user_input])[0]
        confidence = prediction_proba.max() * 100  # highest probability

        # Training accuracy
        training_accuracy = rfModel.score(X_train, y_train) * 100

        # Log prediction
        log_prediction(selected_symptoms, prediction, confidence)

        # Check for simple drift
        simple_drift_check(selected_symptoms, training_avg_symptoms)

        # --- Display Results ---
        st.markdown("---")
        st.success(f"### 🩺 Diagnosis Suggestion: **{prediction}**")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.info(f"**Model Training Accuracy:** {training_accuracy:.2f}%")

        # Retrieve extended info
        match = data_symptoms[data_symptoms['Name'].str.contains(prediction, case=False, na=False)]
        if not match.empty:
            symptoms_text = match['Symptoms'].values[0]
            treatments_text = match['Treatments'].values[0]
            
            with st.expander("🔬 View Associated Symptoms"):
                st.markdown(f"<div style='line-height: 1.8;'>{symptoms_text}</div>", unsafe_allow_html=True)
            
            with st.expander("💊 View Recommended Treatments"):
                st.markdown(f"<div style='line-height: 1.8;'>{treatments_text}</div>", unsafe_allow_html=True)

            # External resources
            query = clean_disease_name(prediction)
            wiki_url = f"https://en.wikipedia.org/wiki/{query.replace('+', '_').title()}"
            mayo_url = f"https://www.mayoclinic.org/search/search-results?q={query}"

            st.markdown("### 🌐 Additional Resources")
            st.markdown(f"- [🔍 Learn more on **Wikipedia**]({wiki_url})")
            st.markdown(f"- [📚 Explore clinical info on **Mayo Clinic**]({mayo_url})")
        else:
            st.warning("No additional information found for this disease.")
