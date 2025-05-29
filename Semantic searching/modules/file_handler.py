import fitz  # PyMuPDF
import nltk
nltk.download('punkt')  # Sentence level text splitter
from nltk.tokenize import sent_tokenize

def chunk_text(text, max_len=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if length + sentence_len <= max_len:
            chunk.append(sentence)
            length += sentence_len
        else:
            chunks.append(" ".join(chunk))
            chunk = sentence.split()[-overlap:]
            length = len(" ".join(chunk))

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def handle_uploads(uploaded_files):
    documents = []
    for file in uploaded_files:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    for chunk in chunk_text(text):
                        documents.append({
                            "text": chunk,
                            "meta": {
                                "filename": file.name,
                                "page": page_num
                            }
                        })
    return documents
