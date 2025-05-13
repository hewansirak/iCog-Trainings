from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from modules.preprocessing import clean_text, expand_synonyms

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_vector_store():
    index = faiss.IndexFlatL2(384)
    return {"index": index, "texts": []}
