"""DeepRetriev -- Streamlit UI for the RAG pipeline."""

import time

import streamlit as st

st.set_page_config(
    page_title="DeepRetriev -- RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("top_k (retrieval depth)", min_value=1, max_value=20, value=5)
    use_hybrid = st.checkbox("Hybrid Search (BM25 + Semantic)", value=True)
    use_reranker = st.checkbox("Cross-Encoder Re-ranker", value=True)

    st.divider()
    st.header("Knowledge Base")

    # Collection stats
    try:
        from src.ingestion.indexer import get_collection_stats

        stats = get_collection_stats()
        st.metric("Indexed chunks", stats["count"])
        st.caption(f"Collection: `{stats['collection']}`")
    except Exception as exc:
        st.warning(f"Could not load collection stats: {exc}")

    # Re-ingest button
    if st.button("Re-ingest corpus"):
        with st.spinner("Loading Wikipedia pages..."):
            from src.ingestion.loader import load_wikipedia_pages

            docs = load_wikipedia_pages()

        with st.spinner(f"Chunking {len(docs)} documents..."):
            from src.ingestion.chunker import chunk_documents

            chunks = chunk_documents(docs)

        with st.spinner(f"Indexing {len(chunks)} chunks..."):
            from src.ingestion.indexer import index_chunks

            index_chunks(chunks, reset=True)

        st.success(f"Ingested {len(chunks)} chunks from {len(docs)} documents.")
        st.rerun()

    st.divider()
    st.header("About")
    st.markdown(
        "A from-scratch RAG pipeline for renewable energy knowledge. "
        "No LangChain. Built as a portfolio project to demonstrate retrieval, "
        "chunking, embedding, re-ranking, and generation with Ollama."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("DeepRetriev -- RAG Pipeline")
st.caption("Ask anything about renewable energy. Answers are grounded in 10 Wikipedia articles.")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input(
    "Your question",
    placeholder="e.g. How does a photovoltaic cell convert sunlight into electricity?",
)

ask = st.button("Ask", type="primary", disabled=not question)

if ask and question:
    from src.config import config
    from src.retrieval.retriever import Retriever

    # Apply sidebar config
    config.retrieval.top_k = top_k
    config.retrieval.use_hybrid = use_hybrid
    config.retrieval.use_reranker = use_reranker

    retriever = Retriever()
    retrieved_chunks = []
    answer_text = ""
    retrieval_ms = 0.0
    generation_ms = 0.0

    # --- Retrieval ---
    try:
        with st.spinner("Retrieving relevant chunks..."):
            t0 = time.perf_counter()
            retrieved_chunks = retriever.retrieve(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1000

        # --- Re-ranking ---
        if use_reranker and retrieved_chunks:
            with st.spinner("Re-ranking with cross-encoder..."):
                from src.retrieval.reranker import Reranker

                reranker = Reranker()
                retrieved_chunks = reranker.rerank(question, retrieved_chunks)

    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # --- Generation ---
    try:
        with st.spinner("Generating answer with Ollama..."):
            from src.generation.generator import generate_answer

            t0 = time.perf_counter()
            answer_text = generate_answer(question, retrieved_chunks)
            generation_ms = (time.perf_counter() - t0) * 1000

    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    total_ms = retrieval_ms + generation_ms

    # Save to session state
    st.session_state.history.append(
        {
            "question": question,
            "answer": answer_text,
            "chunks": retrieved_chunks,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        }
    )

# ---------------------------------------------------------------------------
# Display results (most recent first)
# ---------------------------------------------------------------------------

for entry in reversed(st.session_state.history):
    st.markdown(f"**Q:** {entry['question']}")

    st.info(entry["answer"])

    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{entry['retrieval_ms']:.0f} ms")
    cols[1].metric("Generation", f"{entry['generation_ms']:.0f} ms")
    cols[2].metric("Total", f"{entry['total_ms']:.0f} ms")

    with st.expander(f"Sources ({len(entry['chunks'])} chunks)"):
        for chunk in entry["chunks"]:
            score_pct = chunk.score * 100 if chunk.score <= 1.0 else chunk.score
            st.markdown(
                f"**{chunk.title}** -- score: {score_pct:.1f}%"
            )
            st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
            if chunk.source.startswith("http"):
                st.markdown(f"[Wikipedia link]({chunk.source})")
            st.divider()

    st.divider()
