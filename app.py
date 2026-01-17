import streamlit as st
from groq import Groq

# Read API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL = "llama3-70b-8192"

st.set_page_config(page_title="Groq LLaMA 3 Chat", layout="centered")
st.title("⚡ Groq + LLaMA 3 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.messages,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
