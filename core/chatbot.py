from openai import OpenAI
from core.embeddings import search_similar
import os

api_key=os.getenv("OPENAI_API_KEY")
if not api_key: raise ValueError("Missing OPENAI_API_KEY")

client=OpenAI(api_key=api_key)
MODEL="gpt-4.1-mini"

TONE = "Sei l'Assistente Istituzionale del Comune di Arezzo. Rispondi in modo formale."

FALLBACK="Informazioni non disponibili. Contattare: https://bit.ly/avviachat"

def answer_question(q):
    docs=search_similar(q)
    if not docs: return FALLBACK
    ctx="\n\n".join([d["text"] for d in docs])
    prompt=f"""{TONE}
Contenuti ufficiali:
{ctx}

Domanda: {q}
Risposta formale:
"""
    try:
        r=client.responses.create(model=MODEL, input=prompt)
        return r.output[0].content[0].text
    except:
        return FALLBACK
