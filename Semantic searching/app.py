import streamlit as st
from modules.file_handler import handle_uploads
from modules.embeddings import get_vector_store, embed_and_store
from modules.search import search_query
from modules.memory import update_memory, get_full_query
from modules.visualization import visualize_embeddings
from modules.qa import answer_question
from modules.file_handler import handle_uploads

st.set_page_config(layout="wide")
st.title("📚 Semantic Search Engine")

if "memory" not in st.session_state:
    st.session_state.memory = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_files = st.file_uploader("Upload files (.pdf or .txt)", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    texts_with_meta = handle_uploads(uploaded_files)
    vector_store = get_vector_store()
    embed_and_store(texts_with_meta, vector_store)
    st.session_state.vector_store = vector_store
    st.success("✅ Files uploaded and indexed!")

style = st.radio("Response Style", ["Concise", "Verbose"], horizontal=True)

query = st.text_input("Ask a question")

if query and st.session_state.vector_store:
    full_query = get_full_query(st.session_state.memory, query)
    results = search_query(full_query, st.session_state.vector_store, style=style)
    st.session_state.memory = update_memory(st.session_state.memory, query)

if "?" in query or query.lower().startswith("what") or "conclusion" in query.lower():
    context = " ".join([r["text"] for r in results])
    answer, score = answer_question(query, context)
    st.success(f"📌 Best Answer: **{answer}** (Confidence: {score * 100:.2f}%)")

    st.write("### 🔍 Top 3 Results:")
    for res in results:
        st.chat_message("assistant").markdown(
            f"📄 **Page {res['meta']['page']}** from `{res['meta'].get('filename', 'Unknown Source')}`:\n> {res['text']}"
        )

    # Show named entities in sidebar
    st.sidebar.markdown("## 🧠 Named Entities Found")
    for res in results:
        ents = res["meta"].get("entities", [])
        if ents:
            st.sidebar.markdown(f"**{res['meta'].get('filename', 'Unknown Source')}** — Page {res['meta'].get('page', '?')}")
            st.sidebar.markdown(", ".join(set(ents)))

    # Extractive QA
    st.write("---")
    if st.checkbox("💡 Ask a follow-up question based on top result"):
        follow_up = st.text_input("Your question")
        if follow_up:
            context = results[0]["text"]
            answer, score = answer_question(follow_up, context)
            st.success(f"🧠 Answer: **{answer}** (Confidence: {score:.2f})")

# Visualization
if st.session_state.vector_store and len(st.session_state.vector_store["texts"]) >= 3:
    with st.expander("📊 Show Embedding Visualization"):
        method = st.selectbox("Choose method", ["tsne", "umap"])
        try:
            fig = visualize_embeddings(st.session_state.vector_store, method=method)
            st.pyplot(fig, use_container_width=False)
        except Exception as e:
            st.error(f"Visualization failed: {e}")
elif st.session_state.vector_store:
    st.warning("Upload more documents for meaningful visualization (at least 3 chunks).")
