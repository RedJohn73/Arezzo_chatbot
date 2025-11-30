from openai import OpenAI
from core.embeddings import search_similar
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = "gpt-4.1-mini"

INSTITUTIONAL_TONE = (
    "Sei l'Assistente Istituzionale del Comune di Arezzo. "
    "Rispondi sempre in modo formale e basato sui contenuti ufficiali."
)

FALLBACK_MESSAGE = (
    "Al momento non risultano disponibili informazioni ufficiali utili. "
    "Può utilizzare la chat WhatsApp: https://bit.ly/avviachat"
)

def answer_question(query):
    docs = search_similar(query)
    if not docs:
        return FALLBACK_MESSAGE
    context = "\n\n".join([d["text"] for d in docs])
    prompt = f"{INSTITUTIONAL_TONE}\n\nContenuti:\n{context}\n\nDomanda: {query}\nRisposta:"
    try:
        r = client.responses.create(model=LLM_MODEL, input=prompt)
        return r.output[0].content[0].text
    except:
        return FALLBACK_MESSAGE
