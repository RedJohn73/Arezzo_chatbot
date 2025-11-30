import streamlit as st
from core.chatbot import answer_question
from core.scraper import incremental_crawl
from core.embeddings import build_embeddings_incremental
from core.pdf_handler import extract_text_from_pdf
import json, os, tempfile

st.set_page_config(page_title="Comune di Arezzo – Chatbot", page_icon="🏛️", layout="wide")

# ----------------------------------------------------------
# INFO COUNTERS – REAL-TIME STATS
# ----------------------------------------------------------
st.sidebar.subheader("📊 Stato Attuale del Knowledge Base")

# Count crawler pages
crawler_count = 0
crawler_path = "data/comune_arezzo_dump.json"
if os.path.exists(crawler_path):
    with open(crawler_path, "r", encoding="utf-8") as f:
        try:
            crawler_count = len(json.load(f))
        except:
            crawler_count = 0

# Count uploaded docs
uploaded_count = 0
uploads_path = "data/uploaded_docs.json"
if os.path.exists(uploads_path):
    with open(uploads_path, "r", encoding="utf-8") as f:
        try:
            uploaded_count = len(json.load(f))
        except:
            uploaded_count = 0

# Count FAISS vector chunks
chunk_count = 0
chunk_map_path = "data/chunk_map.json"
if os.path.exists(chunk_map_path):
    with open(chunk_map_path, "r", encoding="utf-8") as f:
        try:
            chunk_map = json.load(f)
            chunk_count = len(chunk_map)
        except:
            chunk_count = 0

# Last embeddings update time
emb_time = "N/D"
if os.path.exists("data/index.faiss"):
    emb_time = str(os.path.getmtime("data/index.faiss"))

st.sidebar.write(f"• **Pagine indicizzate dal crawler:** {crawler_count}")
st.sidebar.write(f"• **Documenti caricati manualmente:** {uploaded_count}")
st.sidebar.write(f"• **Chunk vettoriali in FAISS:** {chunk_count}")
st.sidebar.write(f"• **Ultimo aggiornamento embeddings:** {emb_time}")

st.sidebar.markdown("---")

# ----------------------------------------------------------
# SIDEBAR – ADMIN PANEL
# ----------------------------------------------------------
st.sidebar.title("🛠️ Admin Panel")

# --- Controls for crawling depth ---
st.sidebar.subheader("🔎 Parametri di crawling")

max_pages = st.sidebar.slider(
    "Numero massimo di pagine da analizzare",
    min_value=50, max_value=1000, value=400, step=50
)

max_depth = st.sidebar.slider(
    "Profondità del crawling (1–6)",
    min_value=1, max_value=6, value=4
)

# --- CRAWLING ---
if st.sidebar.button("🔄 Aggiorna contenuti (incrementale)"):
    with st.spinner("Analisi delle pagine nuove o modificate..."):
        updated_docs = incremental_crawl(
            max_pages=max_pages,
            max_depth=max_depth
        )
        if updated_docs == 0:
            st.sidebar.success("Nessuna pagina nuova o aggiornata.")
        else:
            st.sidebar.success(f"{updated_docs} nuove pagine aggiornate!")

    with st.spinner("Aggiornamento degli embeddings..."):
        build_embeddings_incremental()
        st.sidebar.success("Embeddings aggiornati.")

st.sidebar.markdown("---")

# --- DOCUMENT UPLOAD ---
st.sidebar.subheader("📄 Carica documenti (txt/pdf)")

uploaded_file = st.sidebar.file_uploader(
    "Aggiungi documenti al knowledge base",
    type=["txt","pdf"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if uploaded_file.type == "text/plain":
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        text = extract_text_from_pdf(tmp_path)

    os.makedirs("data", exist_ok=True)

    doc_path = "data/uploaded_docs.json"
    docs = []

    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

    docs.append({"source": uploaded_file.name, "text": text})

    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    st.sidebar.success(f"{uploaded_file.name} caricato, sarà incluso nei prossimi embeddings.")

st.sidebar.markdown("---")

# ----------------------------------------------------------
# CHAT – MAIN AREA
# ----------------------------------------------------------
st.markdown("<h1 style='text-align:center'>🏛️ Assistente Istituzionale – Comune di Arezzo</h1>", unsafe_allow_html=True)

# --- CLEAR CHAT BUTTON (under prompt, small) ---
clear_col = st.columns([5,1,5])[1]
with clear_col:
    if st.button("🧹", help="Pulisci chat", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# Chat history init
if "history" not in st.session_state:
    st.session_state["history"] = []

# Render chat messages
for u, b in st.session_state["history"]:
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(b)

# Input box
prompt = st.chat_input("Scriva la sua richiesta...")

if prompt:
    st.chat_message("user").write(prompt)
    response = answer_question(prompt)
    st.chat_message("assistant").write(response)
    st.session_state["history"].append((prompt, response))
