import streamlit as st
import pandas as pd
import joblib
import re
import os
from datetime import datetime

# --- Load assets ---
rfModel = joblib.load('rf_model.pkl')
data = pd.read_csv('training_data.csv')
data_symptoms = pd.read_csv('Diseases_Symptoms.csv')
all_symptoms = data.drop(columns=['prognosis', 'Unnamed: 133'], errors='ignore').columns.tolist()

# --- Prepare training info for accuracy & drift ---
X_train = data.drop(columns=['prognosis', 'Unnamed: 133'], errors='ignore')
y_train = data['prognosis']
training_avg_symptoms = X_train.sum(axis=1).mean()

# --- Page config & navigation ---
st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Disease Predictor", "Monitoring Dashboard"])

# --- Global style ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
    h1, h3 { color: #2c3e50; }
    .stButton>button { background-color: #00c8b4; color: white; font-weight: bold; }
    .stButton>button:hover { background-color: #00b2a1; }
    </style>
""", unsafe_allow_html=True)

# --- Monitoring functions ---
def log_prediction(selected_symptoms, prediction, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms_selected": len(selected_symptoms),
        "prediction": prediction,
        "confidence": confidence
    }
    df = pd.DataFrame([entry])
    if not os.path.isfile('monitoring_log.csv'):
        df.to_csv('monitoring_log.csv', index=False)
    else:
        df.to_csv('monitoring_log.csv', mode='a', header=False, index=False)

def simple_drift_check(selected_symptoms):
    count = len(selected_symptoms)
    if abs(count - training_avg_symptoms) > 0.5 * training_avg_symptoms:
        st.warning("⚠️ Symptom selection pattern looks unusual compared to training data. Possible data drift.")

# --- Helper functions ---
def create_input_vector(selected_symptoms, all_symptoms):
    return [1 if s in selected_symptoms else 0 for s in all_symptoms]

def clean_disease_name(name):
    n = re.sub(r'\(.*?\)', '', name)
    n = re.sub(r'[^a-zA-Z0-9\s]', '', n).strip().replace(" ", "+")
    return n.lower()

# --- Page: Disease Predictor ---
if page == "Disease Predictor":
    st.markdown("# 🧠 AI-Powered Disease Predictor")
    st.markdown("Use intelligent symptom analysis to identify likely diseases and get expert-based treatment recommendations.")
    st.markdown("### 👉 Select your current symptoms:")
    selected = []
    cols = st.columns(4)
    for i, sym in enumerate(all_symptoms):
        label = sym.replace('_', ' ').capitalize()
        if cols[i % 4].checkbox(label):
            selected.append(sym)

    if st.button("🔍 Analyze Symptoms & Predict"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            if len(selected) < 3:
                st.info("💡 Tip: Selecting at least 3 symptoms usually improves confidence.")

            vec = create_input_vector(selected, all_symptoms)
            pred = rfModel.predict([vec])[0]
            proba = rfModel.predict_proba([vec])[0]
            conf = proba.max() * 100
            train_acc = rfModel.score(X_train, y_train) * 100

            log_prediction(selected, pred, conf)
            simple_drift_check(selected)

            st.markdown("---")
            st.success(f"### 🩺 Diagnosis: **{pred}**")
            st.markdown("### 🔵 Confidence Level")
            st.progress(conf / 100)
            st.info(f"**Confidence:** {conf:.2f}%")
            st.info(f"**Training Accuracy:** {train_acc:.2f}%")
            if conf < 30:
                st.warning("⚠️ Low confidence. Please double-check symptoms or consult a professional.")

            match = data_symptoms[data_symptoms['Name'].str.contains(pred, case=False, na=False)]
            if not match.empty:
                with st.expander("🔬 View Associated Symptoms"):
                    st.markdown(match['Symptoms'].values[0], unsafe_allow_html=True)
                with st.expander("💊 View Recommended Treatments"):
                    st.markdown(match['Treatments'].values[0], unsafe_allow_html=True)
                q = clean_disease_name(pred)
                st.markdown("### 🌐 Additional Resources")
                st.markdown(f"- [🔍 Wikipedia](https://en.wikipedia.org/wiki/{q.replace('+','_').title()})")
                st.markdown(f"- [📚 Mayo Clinic](https://www.mayoclinic.org/search/search-results?q={q})")
            else:
                st.warning("No additional info found.")

# --- Page: Monitoring Dashboard ---
elif page == "Monitoring Dashboard":
    st.title("📊 Monitoring Dashboard")
    st.markdown("Analyze app performance over time.")
    if os.path.exists('monitoring_log.csv'):
        df = pd.read_csv('monitoring_log.csv', parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        st.subheader("🕓 Confidence Over Time")
        st.line_chart(df['confidence'])

        st.subheader("🩺 Symptoms Selected Over Time")
        st.area_chart(df['symptoms_selected'])

        avg_conf = df['confidence'].mean()
        avg_sym = df['symptoms_selected'].mean()
        c1, c2 = st.columns(2)
        c1.metric("📈 Avg Confidence", f"{avg_conf:.2f}%")
        c2.metric("🩻 Avg Symptoms Selected", f"{avg_sym:.2f}")

        st.subheader("⚡ Drift Detection Summary")
        recent = df.tail(20)['symptoms_selected'].mean()
        if recent > training_avg_symptoms * 1.5 or recent < training_avg_symptoms * 0.5:
            st.error("Recent symptom selections show strong drift!")
        else:
            st.success("No major drift detected.")

        st.markdown("---")
        st.download_button(
            "📥 Download Monitoring Log",
            df.to_csv(index=True),
            file_name="monitoring_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No monitoring data yet. Run a prediction first.")
