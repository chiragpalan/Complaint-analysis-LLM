import streamlit as st
from groq import Groq
import json

# Initialize Groq client using Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Banking Complaint Classifier", layout="centered")
st.title("🏦 Banking Complaint Classifier")

st.write("Enter a customer complaint below and classify it automatically.")

# Input box
complaint_text = st.text_area(
    "Customer Complaint",
    placeholder="Type or paste the customer complaint here...",
    height=150
)

# Submit button
submit = st.button("Classify Complaint")

if submit:
    if not complaint_text.strip():
        st.warning("Please enter a complaint before submitting.")
    else:
        with st.spinner("Classifying complaint..."):
            prompt = f"""
You are a banking complaint classification system.

Analyze the complaint and return ONLY valid JSON with the following fields:
- category
- sub_category
- product_or_service
- urgency (Low, Medium, High)
- confidence (0 to 1)

Complaint:
\"\"\"{complaint_text}\"\"\"
"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert banking domain analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            raw_output = response.choices[0].message.content

            try:
                result = json.loads(raw_output)
            except json.JSONDecodeError:
                st.error("Model returned invalid JSON. Try again.")
                st.code(raw_output)
                st.stop()

        # Output boxes
        st.subheader("📊 Classification Result")

        col1, col2 = st.columns(2)
        col1.text_input("Category", result.get("category", ""), disabled=True)
        col2.text_input("Sub-category", result.get("sub_category", ""), disabled=True)

        col3, col4 = st.columns(2)
        col3.text_input("Product / Service", result.get("product_or_service", ""), disabled=True)
        col4.text_input("Urgency", result.get("urgency", ""), disabled=True)

        st.text_input("Confidence Score", str(result.get("confidence", "")), disabled=True)
