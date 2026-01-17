import streamlit as st
from groq import Groq
import json
import re

# -------------------------------
# Helper: Robust JSON extraction
# -------------------------------
def extract_json(text: str):
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
# Custom CSS for better fonts and styling
# -------------------------------
st.markdown(
    """
    <style>
    /* Set custom font for whole app */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    /* Input box styling */
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    /* Cards for outputs */
    .output-card {
        background-color: #f7f9fc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Banking Complaint Classifier", layout="wide")

st.markdown("<h1 style='text-align:center'>🏦 Banking Complaint Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:18px'>Paste a customer complaint to classify, summarize and detect the theme.</p>", unsafe_allow_html=True)

# Input box in centered layout
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

        # Classification Fields
        st.markdown("<div class='output-card'><b>Category:</b> {}</div>".format(result.get("category","")), unsafe_allow_html=True)
        st.markdown("<div class='output-card'><b>Sub-category:</b> {}</div>".format(result.get("sub_category","")), unsafe_allow_html=True)
        st.markdown("<div class='output-card'><b>Product / Service:</b> {}</div>".format(result.get("product_or_service","")), unsafe_allow_html=True)
        st.markdown("<div class='output-card'><b>Urgency:</b> {}</div>".format(result.get("urgency","")), unsafe_allow_html=True)
        st.markdown("<div class='output-card'><b>Confidence Score:</b> {}</div>".format(result.get("confidence","")), unsafe_allow_html=True)

        # Summary
        st.markdown("<div class='output-card'><b>Summary:</b><br>{}</div>".format(result.get("summary","")), unsafe_allow_html=True)

        # Theme
        st.markdown("<div class='output-card'><b>Theme:</b> {}</div>".format(result.get("theme","")), unsafe_allow_html=True)
