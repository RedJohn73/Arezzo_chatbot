import streamlit as st
from core.chatbot import answer_question
from core.scraper import crawl_comune_arezzo
from core.embeddings import build_embeddings
import json, os

st.set_page_config(page_title="Municipality of Arezzo Chatbot", page_icon="🏛️", layout="wide")

st.sidebar.title("Admin Panel")
if st.sidebar.button("🔄 Scrape Website"):
    with st.spinner("Scraping..."):
        data = crawl_comune_arezzo()
        os.makedirs("data", exist_ok=True)
        with open("data/comune_arezzo_dump.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        build_embeddings("data/comune_arezzo_dump.json")
        st.sidebar.success("Done!")

uploaded = st.sidebar.file_uploader("Upload documents", type=["txt","pdf","md"])

st.markdown("<h1 style='text-align:center'>🏛️ Municipality of Arezzo Chatbot</h1>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

for u,b in st.session_state["history"]:
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(b)

prompt = st.chat_input("Ask a question...")

if prompt:
    st.chat_message("user").write(prompt)
    response = answer_question(prompt)
    st.chat_message("assistant").write(response)
    st.session_state["history"].append((prompt, response))
