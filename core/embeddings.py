import faiss, json, numpy as np

INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.json"

def build_embeddings(json_path):
    with open(json_path) as f: data = json.load(f)
    texts = [d["text"] for d in data]
    vecs = np.random.rand(len(texts), 768).astype("float32")  # dummy
    index = faiss.IndexFlatL2(768)
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w") as f: json.dump(data, f)

def search_similar(query):
    try:
        index = faiss.read_index(INDEX_PATH)
    except:
        return []
    qv = np.random.rand(1,768).astype("float32")
    D,I = index.search(qv, 3)
    with open(DOCS_PATH) as f: docs = json.load(f)
    return [docs[i] for i in I[0] if i < len(docs)]
