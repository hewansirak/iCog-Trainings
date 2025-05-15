from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from modules.preprocessing import clean_text, expand_synonyms, extract_entities

model = SentenceTransformer('all-mpnet-base-v2')

def get_vector_store():
    # 384 is the dimension for all-mpnet-base-v2
    index = faiss.IndexFlatL2(768)
    return {"index": index, "texts": []}

def embed_and_store(texts, vector_store):
    for t in texts:
        text_clean = clean_text(t["text"])
        text_with_synonyms = expand_synonyms(text_clean)
        embedding = model.encode(text_with_synonyms)

        entities = extract_entities(text_clean)
        meta = t.get("meta", {})
        meta["entities"] = entities

        vector_store["index"].add(np.array([embedding]).astype("float32"))
        vector_store["texts"].append({
            "text": text_clean,
            "meta": meta
        })

    return vector_store
