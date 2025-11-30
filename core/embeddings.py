# core/embeddings.py
# Incremental embedding engine for the Arezzo municipal chatbot

from openai import OpenAI
import faiss
import numpy as np
import json
import os
import tiktoken

# -----------------------------
# CONFIG
# -----------------------------
INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.json"
CHUNK_MAP_PATH = "data/chunk_map.json"  # mapping chunk-id -> doc-id

MAX_TOKENS_PER_CHUNK = 6000  # safe for text-embedding-3-large

# -----------------------------
# INIT
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
ENC = tiktoken.get_encoding("cl100k_base")


# ============================================================
# 1. CHUNKING
# ============================================================
def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    """Splits text into safe chunks for embedding."""
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


# ============================================================
# 2. EMBEDDING
# ============================================================
def embed(texts):
    """Embeds a list of texts using OpenAI embeddings."""
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([d.embedding for d in resp.data]).astype("float32")


# ============================================================
# 3. LOADING EXISTING INDEX OR CREATING NEW ONE
# ============================================================
def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


# ============================================================
# 4. LOAD/SAVE DOCS AND CHUNK MAP
# ============================================================
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


# ============================================================
# 5. INCREMENTAL EMBEDDING REBUILD
# ============================================================
def build_embeddings_incremental():
    """
    Rebuild embeddings incrementally:
    - Only embed new or modified documents.
    - Merge new chunks into existing FAISS index.
    """

    # Load existing docs & chunk map
    existing_docs = load_docs()
    existing_chunk_map = load_chunk_map()
    existing_index = load_index()

    # Load fresh crawler output (which overwrites data/comune_arezzo_dump.json)
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

    # Merge new docs (crawler + uploads)
    new_docs = crawler_docs + uploaded_docs

    # Identify NEW documents (not in existing set)
    to_embed_docs = []
    for d in new_docs:
        if d not in existing_docs:
            to_embed_docs.append(d)

    # If nothing to embed → nothing to do
    if not to_embed_docs:
        print("No new documents to embed.")
        return

    # ------------------------------------
    # CHUNK + EMBED new documents
    # ------------------------------------
    all_new_vectors = []
    new_chunk_map_entries = []

    for doc in to_embed_docs:
        enriched = (
            f"{doc.get('title','')} "
            f\"{' > '.join(doc.get('breadcrumbs', []))}\" "
            f"{doc.get('meta_description','')} "
            f"{doc.get('meta_keywords','')} "
            f"{doc['text']}"
        )

        chunks = chunk_text(enriched)
        vectors = embed(chunks)  # embed chunks

        # append vector chunks
        for vec in vectors:
            all_new_vectors.append(vec)
            new_chunk_map_entries.append(len(existing_docs))  # doc-id = index in saved docs

        # add doc to existing docs
        existing_docs.append(doc)

    # ------------------------------------
    # MERGE INTO FAISS INDEX
    # ------------------------------------
    all_new_vectors = np.vstack(all_new_vectors)

    if existing_index is None:
        # Create new FAISS index
        index = faiss.IndexFlatL2(all_new_vectors.shape[1])
        index.add(all_new_vectors)
    else:
        # Merge with existing
        index = existing_index
        index.add(all_new_vectors)

    # Update chunk map
    existing_chunk_map.extend(new_chunk_map_entries)

    # Save everything
    save_index(index)
    save_docs(existing_docs)
    save_chunk_map(existing_chunk_map)

    print(f"Added {len(new_chunk_map_entries)} vector chunks to FAISS index.")


# ============================================================
# 6. SEMANTIC SEARCH
# ============================================================
def search_similar(query, top_k=5):
    """Search in FAISS and return top-k documents (not chunks)."""
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
