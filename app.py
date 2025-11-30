import streamlit as st
from core.chatbot import answer_question
from core.scraper import incremental_crawl
from core.embeddings import build_embeddings_incremental
from core.pdf_handler import extract_text_from_pdf
import json, os, tempfile

st.set_page_config(page_title="Comune di Arezzo – Chatbot", page_icon="🏛️", layout="wide")

# -----------------------------------------
# SIDEBAR – ADMIN PANEL
# -----------------------------------------
st.sidebar.title("🛠️ Admin Panel")

st.sidebar.subheader("Crawling del sito")

if st.sidebar.button("🔄 Aggiorna contenuti (incrementale)"):
    with st.spinner("Analisi pagine nuove o modificate..."):
        updated_docs = incremental_crawl()
        if updated_docs == 0:
            st.sidebar.success("Nessuna pagina nuova o modificata.")
        else:
            st.sidebar.success(f"{updated_docs} pagine nuove o aggiornate!")

    with st.spinner("Aggiornamento embeddings..."):
        build_embeddings_incremental()
        st.sidebar.success("Embeddings aggiornati.")

st.sidebar.markdown("---")

# -----------------------------------------
# UPLOAD DOCUMENTI
# -----------------------------------------
st.sidebar.subheader("📄 Aggiungi documenti")

uploaded_file = st.sidebar.file_uploader("Carica documento (.txt o .pdf)", type=["txt","pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if uploaded_file.type == "text/plain":
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        text = extract_text_from_pdf(tmp_path)

    # append document to data/documents.json
    os.makedirs("data", exist_ok=True)
    doc_path = "data/uploaded_docs.json"
    docs = []

    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

    docs.append({"source": uploaded_file.name, "text": text})
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    st.sidebar.success(f"{uploaded_file.name} caricato con successo! Sarà incluso nei nuovi embeddings.")

st.sidebar.markdown("---")

# -----------------------------------------
# CHATBOT INTERFACE
# -----------------------------------------
st.markdown("<h1 style='text-align:center'>🏛️ Assistente Istituzionale – Comune di Arezzo</h1>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

for u, b in st.session_state["history"]:
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(b)

prompt = st.chat_input("Scriva la sua richiesta ...")

if prompt:
    st.chat_message("user").write(prompt)
    res = answer_question(prompt)
    st.chat_message("assistant").write(res)
    st.session_state["history"].append((prompt, res))
