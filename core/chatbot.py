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
    """
    q = domanda attuale
    history = lista (utente, bot)
    """

    # Recupero documenti simili dal DB
    docs = search_similar(q)
    context = "\n\n".join([d["text"] for d in docs]) if docs else ""

    # Conversazione precedente
    conv = ""
    if history:
        for u, b in history:
            conv += f"Utente: {u}\nAssistente: {b}\n"

    prompt = f"""
{TONE}

Conversazione precedente:
{conv}

Contenuti ufficiali disponibili:
{context}

Domanda attuale: {q}

Rispondi in modo istituzionale e solo sulla base delle fonti ufficiali.
"""

    try:
        r = client.responses.create(model=MODEL, input=prompt)
        return r.output[0].content[0].text
    except:
        return FALLBACK
