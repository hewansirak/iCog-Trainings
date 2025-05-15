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
    # st.json(vector_store["texts"][0])
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

# Ensure vector_store and results exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "top_result" not in st.session_state:
    st.session_state.top_result = None

# Move Follow-up Q&A above the visualization
st.write("---")
st.markdown("### 💬 Follow-up Q&A")

if st.checkbox("🧠 Enable chat mode") and st.session_state.top_result:
    # Initialize chat
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I'm ready for follow-up questions. Ask anything based on the top result!"
        })

    # Display chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input for follow-up question
    follow_up = st.chat_input("Ask a follow-up question (or type 'exit' to stop)...")

    if follow_up:
        if follow_up.strip().lower() == "exit":
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "✅ Ending chat. Let me know if you need anything else!"
            })
        else:
            st.session_state.chat_history.append({"role": "user", "content": follow_up})
            top_context = st.session_state.top_result["text"]
            follow_up_answer, follow_up_score = answer_question(follow_up, top_context)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Answer**: {follow_up_answer}  \n_Confidence: {follow_up_score * 100:.2f}%_"
            })

# Store top result for follow-up context
if query and st.session_state.vector_store and results:
    st.session_state.top_result = results[0]

# Visualization after Q&A
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
