from openai import OpenAI
from core.embeddings import search_similar
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini"

TONE = (
    "Sei l'Assistente Istituzionale del Comune di Arezzo. "
    "Rispondi sempre in modo formale, chiaro e conforme al linguaggio della PA."
)

FALLBACK = (
    "Al momento non risultano informazioni ufficiali sufficienti. "
    "È possibile contattare la chat WhatsApp del Comune: https://bit.ly/avviachat"
)

def answer_question(q, history=None):
    docs = search_similar(q)
    context = "\n\n".join([d["text"] for d in docs]) if docs else ""

    # Conversazione precedente
    conv = ""
    if history:
        for u, b in history[-6:]:   # LIMITIAMO AGLI ULTIMI 6 TURNI
            conv += f"Utente: {u}\nAssistente: {b}\n"

    prompt = f"""
{TONE}

Contenuti ufficiali:
{context}

Conversazione precedente:
{conv}

Domanda attuale: {q}

Rispondi SOLO sulla base dei contenuti ufficiali.
Se una informazione NON è presente nelle fonti, rispondi:
"Al momento non risultano disponibili informazioni ufficiali utili per rispondere alla sua richiesta. "
    "Può utilizzare la chat WhatsApp del Comune di Arezzo negli orari di apertura degli uffici "
    "per ottenere assistenza diretta tramite il seguente link: https://bit.ly/avviachat"
"""
    try:
        r = client.responses.create(model=MODEL, input=prompt)
        return r.output[0].content[0].text
    except:
        return FALLBACK
