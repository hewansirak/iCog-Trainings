import numpy as np
from modules.embeddings import model
from modules.preprocessing import expand_synonyms

def search_query(query, store, top_k=3):
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
            short_text = doc["text"]
            # Keep just the first sentence or limit to N characters
            short_text = short_text.split(".")[0] + "." if "." in short_text else short_text
            short_text = short_text[:250] + "..." if len(short_text) > 250 else short_text
            results.append({
                "text": short_text,
                "meta": doc["meta"]
            })
    return results
