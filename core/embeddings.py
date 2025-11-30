# core/embeddings.py
# Incremental embedding engine for the Arezzo municipal chatbot
# Includes:
# - Token-safe chunking
# - Incremental FAISS vector index
# - Integration of crawler docs + uploaded docs
# - OpenAI embeddings (text-embedding-3-large)

from openai import OpenAI
import faiss
import numpy as np
import json
import os
import tiktoken

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.json"
CHUNK_MAP_PATH = "data/chunk_map.json"

MAX_TOKENS_PER_CHUNK = 6000  # safe for text-embedding-3-large

# -------------------------------------------------
# INIT
# -------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
ENC = tiktoken.get_encoding("cl100k_base")


# =================================================
# 1. TEXT CHUNKING
# =================================================
def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    """Split text into token-safe chunks."""
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = ENC.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks


# =================================================
# 2. OPENAI EMBEDDING
# =================================================
def embed(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([e.embedding for e in resp.data]).astype("float32")


# =================================================
# 3. LOAD/SAVE HELPERS
# =================================================
def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


def load_docs():
    if not os.path.exists(DOCS_PATH):
        return []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_docs(docs):
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def load_chunk_map():
    if not os.path.exists(CHUNK_MAP_PATH):
        return []
    with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunk_map(chunk_map):
    with open(CHUNK_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)


# =================================================
# 4. INCREMENTAL EMBEDDING PIPELINE
# =================================================
def build_embeddings_incremental():
    """
    Update embeddings WITHOUT recomputing everything.
    - Loads existing docs + FAISS index
    - Loads newly scraped docs + uploaded docs
    - Embeds only NEW/UPDATED documents
    - Merges new vectors incrementally into FAISS
    """

    # ---- LOAD EXISTING DATA ----
    existing_docs = load_docs()
    chunk_map = load_chunk_map()
    index = load_index()

    # ---- LOAD NEW SCRAPER OUTPUT ----
    crawler_path = "data/comune_arezzo_dump.json"
    upload_path = "data/uploaded_docs.json"

    crawler_docs = []
    if os.path.exists(crawler_path):
        with open(crawler_path, "r", encoding="utf-8") as f:
            crawler_docs = json.load(f)

    uploaded_docs = []
    if os.path.exists(upload_path):
        with open(upload_path, "r", encoding="utf-8") as f:
            uploaded_docs = json.load(f)

    new_docs_all = crawler_docs + uploaded_docs

    # ---- FIND NEW DOCUMENTS ----
    to_embed_docs = []
    for d in new_docs_all:
        if d not in existing_docs:
            to_embed_docs.append(d)

    if not to_embed_docs:
        print("No new documents to embed.")
        return

    # =================================================
    #  EMBED NEW DOCUMENTS
    # =================================================
    new_vectors = []
    new_chunk_map_entries = []

    for doc in to_embed_docs:

        breadcrumbs_str = " > ".join(doc.get("breadcrumbs", []))

        enriched = (
            f"{doc.get('title', '')} "
            f"{breadcrumbs_str} "
            f"{doc.get('meta_description', '')} "
            f"{doc.get('meta_keywords', '')} "
            f"{doc['text']}"
        )

        chunks = chunk_text(enriched)
        vectors = embed(chunks)

        # Append vectors
        for v in vectors:
            new_vectors.append(v)
            new_chunk_map_entries.append(len(existing_docs))  # doc index

        # Add doc to existing set
        existing_docs.append(doc)

    new_vectors = np.vstack(new_vectors)

    # =================================================
    # MERGE INTO FAISS
    # =================================================
    if index is None:
        # New index
        index = faiss.IndexFlatL2(new_vectors.shape[1])
        index.add(new_vectors)
    else:
        index.add(new_vectors)

    # Update chunk map
    chunk_map.extend(new_chunk_map_entries)

    # ---- SAVE EVERYTHING ----
    save_index(index)
    save_docs(existing_docs)
    save_chunk_map(chunk_map)

    print(f"Embedded {len(new_chunk_map_entries)} new chunks.")


# =================================================
# 5. SEMANTIC SEARCH
# =================================================
def search_similar(query, top_k=5):
    index = load_index()
    if index is None:
        return []

    qvec = embed([query])
    D, I = index.search(qvec, top_k)

    docs = load_docs()
    chunk_map = load_chunk_map()

    results = []
    for idx in I[0]:
        if idx < len(chunk_map):
            doc_id = chunk_map[idx]
            if doc_id < len(docs):
                results.append(docs[doc_id])

    return results
