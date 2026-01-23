import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spellchecker import SpellChecker
from spacy.cli import download

# --------------------------------------------------
# Page config MUST be first Streamlit command
# --------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    layout="wide"
)

# --------------------------------------------------
# Load NLP & Spellchecker
# --------------------------------------------------
@st.cache_resource
def load_nlp():
    try : 
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess 
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
    

nlp = load_nlp()
spell = SpellChecker()


# --------------------------------------------------
# Spell correction
# --------------------------------------------------
def autocorrect_text(text):
    words = []
    for w in text.split():
        if w.isalpha() and len(w) > 2:
            words.append(spell.correction(w) or w)
        else:
            words.append(w)
    return " ".join(words)

# --------------------------------------------------
# Risk keyword mapping
# --------------------------------------------------
RISK_MAPPING = {
    "high": 1.0,
    "weak": 1.0,
    "poor": 1.0,
    "low": 1.0,
    "bad": 1.0,

    "moderate": 0.5,
    "average": 0.5,

    "strong": 0.0,
    "good": 0.0,
    "stable": 0.0
}

# --------------------------------------------------
# Convert text → model features
# --------------------------------------------------
def text_to_features(text):
    text = text.lower()
    doc = nlp(text)

    features = {
        "industrial_risk": 0.5,
        "management_risk": 0.5,
        "financial_flexibility": 0.5,
        "credibility": 0.5,
        "competitiveness": 0.5,
        "operating_risk": 0.5
    }

    for token in doc:
        if token.lemma_ in RISK_MAPPING:
            value = RISK_MAPPING[token.lemma_]

            if "management" in text:
                features["management_risk"] = value
            if "financial" in text or "liquidity" in text or "debt" in text:
                features["financial_flexibility"] = value
            if "operate" in text or "operation" in text:
                features["operating_risk"] = value
            if "industry" in text or "sector" in text or "market" in text:
                features["industrial_risk"] = value
            if "credibility" in text or "reputation" in text:
                features["credibility"] = value
            if "competitive" in text or "competition" in text:
                features["competitiveness"] = value

    return features

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("🏦 Bankruptcy Prediction Dashboard")
st.write(
    "You can either **manually select risk levels** or **describe the company in text**. "
    "AI will auto-fill the risk factors, which you can review and adjust."
)

st.info(
    "💡 Tip: Mention areas like **management**, **financial strength**, "
    "**operations**, **industry**, or **competitiveness**."
)

# --------------------------------------------------
# Session state defaults
# --------------------------------------------------
for key in [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]:
    if key not in st.session_state:
        st.session_state[key] = 0.5

# --------------------------------------------------
# NLP text input
# --------------------------------------------------
st.divider()
st.header("🧠 Describe the Company (AI-assisted)")

user_text = st.text_area(
    "Example: Weak management and low competitiveness, but strong financial flexibility."
)

if user_text.strip():
    corrected_text = autocorrect_text(user_text)

    if corrected_text.lower() != user_text.lower():
        st.markdown(f"ℹ️ Did you mean: *{corrected_text}*")

    extracted = text_to_features(corrected_text)

    st.subheader("🤖 AI-Suggested Risk Levels")
    for k, v in extracted.items():
        st.write(f"- {k.replace('_', ' ').title()}: {v}")

    if st.button("Apply AI Suggestions"):
        for k, v in extracted.items():
            st.session_state[k] = v
        st.success("✅ Risk factors updated. You can adjust them below.")

# --------------------------------------------------
# Manual / Auto-filled Inputs
# --------------------------------------------------
st.divider()
st.header("📝 Review / Adjust Risk Factors")

col1, col2, col3 = st.columns(3)

with col1:
    industrial_risk = st.selectbox(
        "Industrial Risk", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["industrial_risk"])
    )
    management_risk = st.selectbox(
        "Management Risk", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["management_risk"])
    )

with col2:
    financial_flexibility = st.selectbox(
        "Financial Flexibility", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["financial_flexibility"])
    )
    credibility = st.selectbox(
        "Credibility", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["credibility"])
    )

with col3:
    competitiveness = st.selectbox(
        "Competitiveness", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["competitiveness"])
    )
    operating_risk = st.selectbox(
        "Operating Risk", [0.0, 0.5, 1.0],
        index=[0.0, 0.5, 1.0].index(st.session_state["operating_risk"])
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
input_data = np.array([[
    industrial_risk,
    management_risk,
    financial_flexibility,
    credibility,
    competitiveness,
    operating_risk
]])

st.divider()

if st.button("🔮 Predict Bankruptcy"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The company is likely to go **BANKRUPT**")

        st.markdown(
            """
            The current risk indicators suggest potential financial distress.
            Factors such as management challenges, limited financial flexibility, or operational issues may be contributing to this outcome.
            """
         )
    else:
        st.success("✅ The company is NOT likely to go bankrupt")

        st.markdown(
       """
        This prediction suggests that the company does not show strong bankruptcy risk signals at the moment.
        Key risk areas such as management, financial flexibility, and operations appear to be within acceptable levels.
      """
        ) 

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probs
        })
        st.bar_chart(prob_df.set_index("Class"))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("🔹 🔹 🔹")
st.markdown("""
    <div class="custom-footer">
        Designed & built by <b>Sanika Sharma</b><br>
        Bankruptcy Risk Prediction • ML + NLP • Streamlit
    </div>
    """,
    unsafe_allow_html=True)
