import fitz  # PyMuPDF
import os

def handle_uploads(uploaded_files):
    all_texts = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            all_texts.extend(read_pdf(file, file.name))
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            all_texts.append({
                "text": text,
                "meta": {"source": file.name, "page": 1}
            })
    return all_texts

def read_pdf(file, filename):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        texts.append({
            "text": text,
            "meta": {"source": filename, "page": i}
        })
    return texts
