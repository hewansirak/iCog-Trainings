import numpy as np
from modules.embeddings import model
from modules.preprocessing import expand_synonyms

def search_query(query, store, top_k=3):
    expanded_query = expand_synonyms(query)
    q_vec = model.encode(expanded_query)
    D, I = store["index"].search(np.array([q_vec]), top_k)

    results = []
    for idx in I[0]:
        if idx < len(store["texts"]):
            results.append(store["texts"][idx])
    return results
