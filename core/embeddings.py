from openai import OpenAI
import faiss, numpy as np, json, os

INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.json"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    return np.array([e.embedding for e in resp.data]).astype("float32")

def build_embeddings(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        docs=json.load(f)

    enriched = []
    for d in docs:
        comp = f"{d.get('title','')} {d.get('breadcrumbs','')} {d.get('meta_description','')} {d.get('meta_keywords','')} {d['text']}"
        enriched.append(comp)

    vecs = embed(enriched)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH,"w",encoding="utf-8") as f:
        json.dump(docs,f,ensure_ascii=False,indent=2)

def search_similar(query):
    index = faiss.read_index(INDEX_PATH)
    qvec = embed([query])
    D,I = index.search(qvec,5)
    with open(DOCS_PATH,"r",encoding="utf-8") as f:
        docs=json.load(f)
    return [docs[i] for i in I[0] if i<len(docs)]
