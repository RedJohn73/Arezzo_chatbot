# core/pdf_handler.py
import fitz  # PyMuPDF

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text
