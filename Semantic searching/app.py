import streamlit as st
from modules.file_handler import handle_uploads
from modules.embeddings import get_vector_store, embed_and_store
from modules.search import search_query
from modules.memory import update_memory, get_full_query

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
    st.success("Files uploaded and indexed!")

query = st.text_input("Ask a question")

if query and st.session_state.vector_store:
    full_query = get_full_query(st.session_state.memory, query)
    results = search_query(full_query, st.session_state.vector_store)
    st.session_state.memory = update_memory(st.session_state.memory, query)

    st.write("### 🔍 Top 3 Results:")
    for rank, res in enumerate(results, 1):
        st.markdown(f"**{rank}.** *Page {res['meta']['page']}* — `{res['meta']['source']}`")
        st.markdown(res["text"])
        st.markdown("---")

