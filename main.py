"""RAGLab CLI entry point.

Usage:
    python main.py ingest
    python main.py query "What are the main types of renewable energy?"
"""

import argparse
import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup — must run before any module import that logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raglab")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load Wikipedia pages, chunk, embed, and store in ChromaDB."""
    from src.ingestion.loader import load_wikipedia_pages
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.indexer import index_chunks, get_collection_stats

    logger.info("=== RAGLab — Ingestion ===")

    documents = load_wikipedia_pages()
    if not documents:
        logger.error("No documents loaded. Check your internet connection or topic list.")
        sys.exit(1)

    chunks = chunk_documents(documents)
    index_chunks(chunks, reset=args.reset)

    stats = get_collection_stats()
    logger.info(
        "Done. Collection '%s' now contains %d chunks.",
        stats["collection"],
        stats["count"],
    )


def cmd_query(args: argparse.Namespace) -> None:
    """Retrieve relevant chunks and generate an answer with Ollama."""
    from src.retrieval.retriever import Retriever
    from src.generation.generator import generate_answer

    question = args.question
    logger.info("=== RAGLab — Query ===")
    logger.info("Question: %s", question)

    retriever = Retriever()
    try:
        chunks = retriever.retrieve(question, top_k=args.top_k)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not chunks:
        print("\nNo relevant context found in the knowledge base for your question.")
        sys.exit(0)

    logger.info("Generating answer with Ollama (%s) ...", __import__("src.config", fromlist=["config"]).config.ollama.model)
    try:
        answer = generate_answer(question, chunks)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    separator = "─" * 72

    print(f"\n{separator}")
    print("ANSWER")
    print(separator)
    print(answer)

    print(f"\n{separator}")
    print("SOURCES")
    print(separator)
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source not in seen:
            print(f"  • {chunk.title}")
            print(f"    {chunk.source}")
            seen.add(chunk.source)

    print(separator)


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server via uvicorn."""
    import uvicorn
    from src.api.app import create_app

    app = create_app()
    logger.info(
        "Starting RAGLab API server on %s:%d", args.host, args.port
    )
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="RAGLab — Retrieval-Augmented Generation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into ChromaDB")
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collection before ingestion",
    )

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="Your question")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Number of context chunks to retrieve (default: from config)",
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "serve": cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
