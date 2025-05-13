import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap

def visualize_embeddings(store, method="tsne"):
    embeddings = np.array([text["embedding"] for text in store["texts"]])
    labels = [text["meta"]["source"] for text in store["texts"]]

    if len(embeddings) < 2:
        raise ValueError("Not enough data points to visualize. Please upload more content.")

    if method == "tsne":
        perplexity = min(30, max(2, len(embeddings) // 2))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    reduced = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="deep", s=100, ax=ax)
    ax.set_title(f"Embedding Visualization ({method.upper()})")
    return fig
