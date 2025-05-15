import numpy as np
from modules.embeddings import model
from modules.preprocessing import expand_synonyms

def search_query(query, store, style="Concise", top_k=3):
    from modules.embeddings import model
    from modules.preprocessing import expand_synonyms
    import numpy as np

    expanded_query = expand_synonyms(query)
    q_vec = model.encode(expanded_query)
    D, I = store["index"].search(np.array([q_vec]), top_k)

    results = []
    for idx in I[0]:
        if idx < len(store["texts"]):
            doc = store["texts"][idx]
            text = doc["text"]
            if style == "Concise":
                text = text.split(".")[0] + "." if "." in text else text
                text = text[:250] + "..." if len(text) > 250 else text
            results.append({
                "text": text,
                "meta": doc["meta"]
            })
    return results