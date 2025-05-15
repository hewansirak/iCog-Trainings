# 📚 Semantic Search Engine

A powerful Streamlit-based app for document search and AI question answering using semantic embeddings and vector similarity. Upload PDF or TXT files and ask questions in natural language — get accurate answers with confidence scores, context, and entity recognition.

---

## 🚀 Features

- 🔍 **Semantic Search** — Upload docs, ask natural-language questions.
- 💬 **AI-Powered Answers** — Get concise or verbose responses with confidence scores.
- 🧠 **Memory-Aware Q&A** — Follow-up Q&A that builds on previous questions.
- 📊 **Embedding Visualization** — Explore document embeddings via t-SNE or UMAP.
- 🧾 **Named Entity Recognition** — Shows people, locations, etc. found in your docs.

---

## 📁 Supported File Formats

- `.pdf`
- `.txt`

---

## 🛠️ How It Works

1. Upload documents.
2. Text is extracted, cleaned, and embedded using `all-mpnet-base-v2`.
3. Embeddings are indexed using FAISS for fast similarity search.
4. Ask a question → Retrieves relevant chunks → Generates an answer with confidence.
5. Follow-up Q&A leverages chat history and top-ranked results.

---

## 🧰 Tech Stack

- `Streamlit` for the web interface
- `sentence-transformers` for generating embeddings
- `FAISS` for vector similarity search
- `scikit-learn`, `UMAP`, `t-SNE` for dimensionality reduction
- `Matplotlib`, `Seaborn` for visualization
- `spaCy` or similar for entity extraction (via `extract_entities`)
---

## ▶️ Getting Started

```bash
# Clone the repo
https://github.com/hewansirak/iCog-Trainings.git
cd semantic-search-engine

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
