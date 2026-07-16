import os
import streamlit as st

from vectorstore import build_vectorstore
from graph import build_graph

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Self-Healing RAG",
    layout="wide"
)

st.title("🤖 Self-Healing RAG Assistant")
st.write("Upload a PDF, DOCX or TXT file and ask questions.")

# ---------------- CREATE UPLOAD FOLDER ----------------

os.makedirs("uploads", exist_ok=True)

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"]
)

file_path = None

if uploaded_file is not None:

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded: {uploaded_file.name}")

# ---------------- QUESTION ----------------

question = st.text_input(
    "Ask your question"
)

# ---------------- BUTTON ----------------

if st.button("Ask"):

    if uploaded_file is None:
        st.warning("Please upload a document.")
        st.stop()

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Building Vector Database..."):

        retriever = build_vectorstore(file_path)

    with st.spinner("Running Self-Healing RAG..."):

        pipeline = build_graph(retriever)

        initial_state = {
            "question": question,
            "retrieved_chunks": [],
            "answer": "",
            "is_grounded": False,
            "retry_count": 0,
            "final_answer": None,
            "critic_report": {}
        }

        result = pipeline.invoke(initial_state)

    # ---------------- FINAL ANSWER ----------------

    st.divider()

    st.header("💬 Final Answer")

    st.write(result.get("final_answer"))

    # ---------------- CRITIC REPORT ----------------

    st.divider()

    st.header("🔍 Critic Report")

    report = result.get("critic_report", {})

    verdict = report.get("verdict", "UNKNOWN")

    if verdict == "GROUNDED":
        st.success("✅ GROUNDED")

    elif verdict == "PARTIALLY_GROUNDED":
        st.warning("⚠️ PARTIALLY GROUNDED")

    else:
        st.error("❌ NOT GROUNDED")

    st.subheader("✅ Supported Statements")

    supported = report.get("supported", [])

    if supported:

        for item in supported:

            st.success(item)

    else:

        st.info("No supported statements found.")

    st.subheader("❌ Unsupported Statements")

    unsupported = report.get("unsupported", [])

    if unsupported:

        for item in unsupported:

            st.error(item)

    else:

        st.success("No unsupported statements.")

    # ---------------- RETRY COUNT ----------------

    st.divider()

    st.header("🔄 Retry Count")

    st.write(result.get("retry_count"))

    # ---------------- RETRIEVED CHUNKS ----------------

    st.divider()

    st.header("📚 Retrieved Chunks")

    chunks = result.get("retrieved_chunks", [])

    if chunks:

        for i, chunk in enumerate(chunks, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(chunk)

    else:

        st.info("No chunks retrieved.")

    # ---------------- SOURCE PREVIEW ----------------

    st.divider()

    st.header("📄 Source Preview")

    if chunks:

        for i, chunk in enumerate(chunks, start=1):

            st.markdown(f"### Source Chunk {i}")

            st.code(chunk)

    # ---------------- DASHBOARD ----------------

    st.divider()

    st.header("📊 Retrieval Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Retrieved Chunks", len(chunks))
        st.metric("Retry Count", result.get("retry_count", 0))

    with col2:
        st.metric("Embedding Model", "MiniLM-L6-v2")
        st.metric("LLM", "Llama-3.1-8b")