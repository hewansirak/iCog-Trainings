import numpy as np
from modules.embeddings import model
from modules.preprocessing import expand_synonyms, highlight_entities

def search_query(query, store, style="Concise", top_k=3):
    expanded_query = expand_synonyms(query)
    q_vec = model.encode(expanded_query)
    D, I = store["index"].search(np.array([q_vec]), top_k + 5)

    results = []
    query_lower = query.lower()

    for rank, idx in enumerate(I[0]):
        if idx < len(store["texts"]):
            doc = store["texts"][idx]
            text = doc["text"]
            if style == "Concise":
                text = text[:500] + "..." if len(text) > 500 else text

            # Calculate a pseudo-score based on FAISS distance
            score = 1 / (D[0][rank] + 1e-5)

            if "entities" in doc["meta"]:
                if any(ent in query_lower for ent in doc["meta"]["entities"]):
                    score *= 1.2  

            text = highlight_entities(text, doc["meta"].get("entities", []))
            results.append((score, {"text": text, "meta": doc["meta"]}))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return [r[1] for r in results]
