import streamlit as st
from groq import Groq
import json
import re

# -------------------------------
# Helper: Robust JSON extraction
# -------------------------------
def extract_json(text: str):
    """
    Extracts the first valid JSON object from a string.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())

# -------------------------------
# Groq client (API key from Streamlit Cloud secrets)
# -------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ✅ Use a currently supported Groq model
MODEL = "llama-3.3-70b-versatile"
# (You can also use: "llama-3.1-8b-instant")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Banking Complaint Classifier",
    layout="centered"
)

st.title("🏦 Banking Complaint Classifier")
st.write(
    "Paste a customer complaint below. "
    "The system will identify category, sub-category, product/service, and urgency."
)

# Input box
complaint_text = st.text_area(
    "Customer Complaint",
    placeholder="Enter the banking complaint here...",
    height=160
)

# Submit button
submit = st.button("Classify Complaint")

# -------------------------------
# On Submit
# -------------------------------
if submit:
    if not complaint_text.strip():
        st.warning("Please enter a complaint before submitting.")
    else:
        with st.spinner("Classifying complaint..."):
            prompt = f"""
You are a banking complaint classification engine.

STRICT RULES:
- Output ONLY a valid JSON object
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include backticks
- Do NOT include any text outside JSON

The JSON schema MUST be exactly:

{{
  "category": "string",
  "sub_category": "string",
  "product_or_service": "string",
  "urgency": "Low | Medium | High",
  "confidence": number
}}

Complaint:
\"\"\"{complaint_text}\"\"\"
"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON-only banking domain classifier."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            raw_output = response.choices[0].message.content

            try:
                result = extract_json(raw_output)
            except Exception:
                st.error("❌ Model returned invalid JSON.")
                st.write("Raw model output (for debugging):")
                st.code(raw_output)
                st.stop()

        # -------------------------------
        # Output Section
        # -------------------------------
        st.subheader("📊 Classification Result")

        col1, col2 = st.columns(2)
        col1.text_input(
            "Category",
            value=result.get("category", ""),
            disabled=True
        )
        col2.text_input(
            "Sub-category",
            value=result.get("sub_category", ""),
            disabled=True
        )

        col3, col4 = st.columns(2)
        col3.text_input(
            "Product / Service",
            value=result.get("product_or_service", ""),
            disabled=True
        )
        col4.text_input(
            "Urgency",
            value=result.get("urgency", ""),
            disabled=True
        )

        st.text_input(
            "Confidence Score",
            value=str(result.get("confidence", "")),
            disabled=True
        )
