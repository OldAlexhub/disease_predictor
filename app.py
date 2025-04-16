import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import re

# Load assets
rfModel = joblib.load('rf_model.pkl')
data = pd.read_csv('training_data.csv')
data_symptoms = pd.read_csv('Diseases_Symptoms.csv')
all_symptoms = data.drop(columns=['prognosis', 'Unnamed: 133'], errors='ignore').columns.tolist()

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

# --- Prediction Logic ---
def create_input_vector(selected_symptoms, all_symptoms):
    return [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

def clean_disease_name(name):
    name = re.sub(r'\(.*?\)', '', name)  # remove anything in parentheses
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # remove punctuation
    name = name.strip().replace(" ", "+")
    return name.lower()

# --- Predict Button ---
if st.button("🔍 Analyze Symptoms & Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom to continue.")
    else:
        user_input = create_input_vector(selected_symptoms, all_symptoms)
        prediction = rfModel.predict([user_input])[0]
        
        st.markdown("---")
        st.success(f"### 🩺 Diagnosis Suggestion: **{prediction}**")

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
