import streamlit as st
from groq import Groq
import json
import re

# -------------------------------
# Helper: Robust JSON extraction
# -------------------------------
def extract_json(text: str):
    """Extracts the first valid JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())

# -------------------------------
# Groq client
# -------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

# -------------------------------
# Custom CSS for styling
# -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    .output-card {
        background-color: #f7f9fc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .urgency-high {background-color:#FFCCCC;}
    .urgency-medium {background-color:#FFF0B3;}
    .urgency-low {background-color:#CCFFCC;}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Banking Complaint Classifier", layout="wide")
st.markdown("<h1 style='text-align:center'>🏦 Banking Complaint Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:18px'>Paste a customer complaint to classify, summarize, and detect the theme.</p>", unsafe_allow_html=True)

# Input box
complaint_text = st.text_area(
    "Enter Customer Complaint",
    placeholder="Type or paste the complaint here...",
    height=150
)

submit = st.button("🚀 Classify Complaint")

# -------------------------------
# On Submit
# -------------------------------
if submit:
    if not complaint_text.strip():
        st.warning("⚠️ Please enter a complaint before submitting.")
    else:
        with st.spinner("Analyzing complaint..."):
            # Prompt for Groq LLM
            prompt = f"""
You are a banking complaint classification engine.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations, markdown, or backticks
- Do NOT include any text outside JSON

JSON Schema:
{{
  "category": "string",
  "sub_category": "string",
  "product_or_service": "string",
  "urgency": "Low | Medium | High",
  "confidence": number,
  "summary": "string (2-3 line summary)",
  "theme": "string (main topic/theme)"
}}

Complaint:
\"\"\"{complaint_text}\"\"\"
"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only banking classifier and summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            raw_output = response.choices[0].message.content

            # Safe JSON parsing
            try:
                result = extract_json(raw_output)
            except Exception:
                st.error("❌ Model returned invalid JSON.")
                st.code(raw_output)
                st.stop()

        # -------------------------------
        # Display results in cards
        # -------------------------------
        st.markdown("<h2 style='text-align:center'>📊 Classification Result</h2>", unsafe_allow_html=True)

        st.markdown(f"<div class='output-card'><b>Category:</b> {result.get('category','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-card'><b>Sub-category:</b> {result.get('sub_category','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-card'><b>Product / Service:</b> {result.get('product_or_service','')}</div>", unsafe_allow_html=True)

        # Urgency with color and tooltip
        urgency_value = result.get('urgency','')
        urgency_class = "urgency-low"
        if urgency_value.lower() == "medium":
            urgency_class = "urgency-medium"
        elif urgency_value.lower() == "high":
            urgency_class = "urgency-high"

        st.markdown(
            f"""<div class='output-card {urgency_class}'>
                <b>Urgency:</b> 
                <span title='Indicates how critical the complaint is: High=Immediate, Medium=Moderate, Low=Can wait'>{urgency_value}</span>
            </div>""",
            unsafe_allow_html=True
        )

        # Confidence score with hover tooltip
        st.markdown(
            f"""<div class='output-card'>
                <b>Confidence Score:</b> 
                <span title="The model's confidence in the classification (0=low, 1=high)">{result.get('confidence','')}</span>
            </div>""",
            unsafe_allow_html=True
        )

        # Summary
        st.markdown(f"<div class='output-card'><b>Summary:</b><br>{result.get('summary','')}</div>", unsafe_allow_html=True)

        # Theme
        st.markdown(f"<div class='output-card'><b>Theme:</b> {result.get('theme','')}</div>", unsafe_allow_html=True)
