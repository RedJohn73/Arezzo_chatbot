import streamlit as st
from core.chatbot import answer_question
from core.scraper import crawl_comune_arezzo
from core.embeddings import build_embeddings
import json, os

st.set_page_config(page_title="Municipality of Arezzo Chatbot", page_icon="🏛️", layout="wide")

st.sidebar.title("Admin Panel – Comune di Arezzo")

max_pages = st.sidebar.slider("Max pagine da crawlare", 50, 800, 400)
max_depth = st.sidebar.slider("Profondità crawling", 1, 6, 3)

if st.sidebar.button("🔄 Avvia crawling avanzato"):
    with st.spinner("Crawling in corso..."):
        data = crawl_comune_arezzo(max_pages=max_pages, max_depth=max_depth,
            content_types=["news","bando","ordinanza","pagina"])
        os.makedirs("data", exist_ok=True)
        with open("data/comune_arezzo_dump.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        build_embeddings("data/comune_arezzo_dump.json")
        st.sidebar.success("Crawling completato!")

st.markdown("<h1 style='text-align:center'>🏛️ Assistente Istituzionale – Comune di Arezzo</h1>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

for u,b in st.session_state["history"]:
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(b)

prompt = st.chat_input("Scriva la sua richiesta ...")

if prompt:
    st.chat_message("user").write(prompt)
    res = answer_question(prompt)
    st.chat_message("assistant").write(res)
    st.session_state["history"].append((prompt,res))
