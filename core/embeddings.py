# core/embeddings.py
from openai import OpenAI
import faiss
import numpy as np
import json
import os
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS_PER_CHUNK = 6000   # safe limit for text-embedding-3-large
INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.json"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# -----------------------------
# TEXT CHUNKER
# -----------------------------
def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK):
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


# -----------------------------
# EMBEDDER
# -----------------------------
def embed(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([e.embedding for e in resp.data]).astype("float32")


# -----------------------------
# BUILD EMBEDDINGS
# -----------------------------
def build_embeddings(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    vectors = []
    chunks_meta = []   # mapping chunk → doc index

    for i, d in enumerate(docs):
        enriched = (
            f"{d.get('title','')} "
            f"{' > '.join(d.get('breadcrumbs', []))} "
            f"{d.get('meta_description','')} "
            f"{d.get('meta_keywords','')} "
            f"{d['text']}"
        )

        # CHUNKING 🔥
        text_chunks = chunk_text(enriched, MAX_TOKENS_PER_CHUNK)

        # embed chunks
        chunk_vectors = embed(text_chunks)

        for vec in chunk_vectors:
            vectors.append(vec)
            chunks_meta.append(i)  # i = index documento

    vectors = np.vstack(vectors)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    # Save mapping (doc index per chunk)
    meta_path = DOCS_PATH.replace(".json", "_chunks.json")
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f)


# -----------------------------
# SEARCH
# -----------------------------
def search_similar(query, top_k=5):
    index = faiss.read_index(INDEX_PATH)
    qvec = embed([query])

    distances, ids = index.search(qvec, top_k)

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    meta_path = DOCS_PATH.replace(".json", "_chunks.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)

    results = []
    for i in ids[0]:
        if i < len(chunks_meta):
            doc_id = chunks_meta[i]
            results.append(docs[doc_id])

    return results
