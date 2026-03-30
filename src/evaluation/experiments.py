"""Embedding model and chunking configuration experiments for RAGLab."""

import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Generator

from tqdm import tqdm

from src.config import config
from src.evaluation.dataset import EvalSample
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.tracking import log_run
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import index_chunks
from src.ingestion.loader import Document, load_wikipedia_pages
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment collection naming
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Convert a string into a safe ChromaDB collection name component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def _experiment_collection_name(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Build a unique ChromaDB collection name for an experiment configuration."""
    return (
        f"raglab_exp_{_safe_name(embedding_model.split('/')[-1])}"
        f"_c{chunk_size}_o{chunk_overlap}"
    )[:63]  # ChromaDB max collection name length


# ---------------------------------------------------------------------------
# Config context manager — temporarily overrides global config values
# ---------------------------------------------------------------------------

@contextmanager
def _override_config(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
) -> Generator[None, None, None]:
    """Temporarily patch the global config for a single experiment run."""
    orig_model = config.embedding.model_name
    orig_collection = config.chroma.collection_name
    orig_chunk_size = config.chunk.size
    orig_chunk_overlap = config.chunk.overlap

    try:
        config.embedding.model_name = embedding_model
        config.chroma.collection_name = collection_name
        config.chunk.size = chunk_size
        config.chunk.overlap = chunk_overlap
        yield
    finally:
        config.embedding.model_name = orig_model
        config.chroma.collection_name = orig_collection
        config.chunk.size = orig_chunk_size
        config.chunk.overlap = orig_chunk_overlap


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    documents: list[Document],
    dataset: list[EvalSample],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int = 5,
    use_llm_judge: bool = False,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run a single experiment configuration end-to-end.

    Steps:
    1. Chunk documents with the given configuration.
    2. Index chunks using the specified embedding model into a dedicated collection.
    3. Evaluate with RAGEvaluator.
    4. Optionally log to MLflow.

    Args:
        documents: Pre-loaded Wikipedia documents.
        dataset: Evaluation samples.
        embedding_model: HuggingFace model name for embeddings.
        chunk_size: Character-level chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        top_k: Number of chunks to retrieve per query.
        use_llm_judge: Whether to use Ollama LLM-as-judge for generation metrics.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Full evaluation results dict plus timing information.
    """
    collection_name = _experiment_collection_name(embedding_model, chunk_size, chunk_overlap)
    run_name = f"{embedding_model.split('/')[-1]}_c{chunk_size}_o{chunk_overlap}"

    logger.info(
        "Starting experiment: model=%s, chunk_size=%d, overlap=%d",
        embedding_model,
        chunk_size,
        chunk_overlap,
    )

    with _override_config(embedding_model, chunk_size, chunk_overlap, collection_name):
        # --- Chunking ---
        t_chunk_start = time.perf_counter()
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
        chunk_time = time.perf_counter() - t_chunk_start
        logger.info("Chunked into %d chunks in %.1fs", len(chunks), chunk_time)

        # --- Indexing ---
        t_index_start = time.perf_counter()
        index_chunks(chunks, reset=True)
        index_time = time.perf_counter() - t_index_start
        logger.info("Indexed %d chunks in %.1fs", len(chunks), index_time)

        # --- Retrieval + Evaluation ---
        retriever = Retriever()
        evaluator = RAGEvaluator(top_k=top_k, use_llm_judge=use_llm_judge)

        t_eval_start = time.perf_counter()
        results = evaluator.evaluate(dataset, retriever)
        eval_time = time.perf_counter() - t_eval_start

    # Augment results with timing
    results["timing"] = {
        "chunk_time_s": round(chunk_time, 2),
        "index_time_s": round(index_time, 2),
        "eval_time_s": round(eval_time, 2),
        "n_chunks": len(chunks),
    }
    results["config"]["collection_name"] = collection_name

    # --- MLflow logging ---
    if log_to_mlflow:
        params = {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "n_chunks": len(chunks),
        }
        metrics = {
            **results["global_metrics"],
            "index_time_s": index_time,
            "chunk_time_s": chunk_time,
            "eval_time_s": eval_time,
        }
        log_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifact_data=results,
        )

    logger.info(
        "Experiment done — recall@%d=%.3f, MRR=%.3f",
        top_k,
        results["global_metrics"].get("recall_at_k") or 0,
        results["global_metrics"].get("mrr") or 0,
    )
    return results


# ---------------------------------------------------------------------------
# Embedding model experiments
# ---------------------------------------------------------------------------

def run_embedding_experiments(
    dataset: list[EvalSample],
    documents: list[Document] | None = None,
    top_k: int = 5,
    use_llm_judge: bool = False,
) -> list[dict[str, Any]]:
    """Compare embedding models with the baseline chunking configuration.

    Models tested (from config.experiment.embedding_models):
      - all-MiniLM-L6-v2  (baseline — fast, small)
      - all-mpnet-base-v2  (higher quality)
      - BAAI/bge-small-en-v1.5  (BGE, good quality/size ratio)

    Args:
        dataset: Evaluation samples.
        documents: Pre-loaded documents. If None, fetches from Wikipedia.
        top_k: Number of chunks to retrieve.
        use_llm_judge: Whether to run LLM-as-judge generation metrics.

    Returns:
        List of result dicts, one per model.
    """
    if documents is None:
        logger.info("Loading Wikipedia documents...")
        documents = load_wikipedia_pages()

    models = config.experiment.embedding_models
    chunk_size = config.chunk.size
    chunk_overlap = config.chunk.overlap

    all_results: list[dict[str, Any]] = []

    for model in tqdm(models, desc="Embedding models", unit="model"):
        result = run_single_experiment(
            documents=documents,
            dataset=dataset,
            embedding_model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            use_llm_judge=use_llm_judge,
        )
        all_results.append(result)

    _print_experiment_summary(all_results, group_by="embedding_model")
    return all_results


# ---------------------------------------------------------------------------
# Chunking configuration experiments
# ---------------------------------------------------------------------------

def run_chunking_experiments(
    dataset: list[EvalSample],
    documents: list[Document] | None = None,
    top_k: int = 5,
    use_llm_judge: bool = False,
) -> list[dict[str, Any]]:
    """Compare chunking configurations using the baseline embedding model.

    Configurations tested (from config.experiment.chunking_configs):
      - 256 chars, 0 overlap
      - 256 chars, 50 overlap
      - 512 chars, 64 overlap  (baseline)
      - 512 chars, 100 overlap
      - 1024 chars, 100 overlap
      - 1024 chars, 200 overlap

    Args:
        dataset: Evaluation samples.
        documents: Pre-loaded documents. If None, fetches from Wikipedia.
        top_k: Number of chunks to retrieve.
        use_llm_judge: Whether to run LLM-as-judge generation metrics.

    Returns:
        List of result dicts, one per chunking config.
    """
    if documents is None:
        logger.info("Loading Wikipedia documents...")
        documents = load_wikipedia_pages()

    embedding_model = config.embedding.model_name
    chunking_configs = config.experiment.chunking_configs

    all_results: list[dict[str, Any]] = []

    for cfg in tqdm(chunking_configs, desc="Chunking configs", unit="config"):
        chunk_size = cfg["chunk_size"]
        chunk_overlap = cfg["chunk_overlap"]
        result = run_single_experiment(
            documents=documents,
            dataset=dataset,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            use_llm_judge=use_llm_judge,
        )
        all_results.append(result)

    _print_experiment_summary(all_results, group_by="chunking")
    return all_results


# ---------------------------------------------------------------------------
# Evaluate current config only
# ---------------------------------------------------------------------------

def run_evaluation_only(
    dataset: list[EvalSample],
    top_k: int = 5,
    use_llm_judge: bool | None = None,
    log_to_mlflow: bool = False,
) -> dict[str, Any]:
    """Evaluate the current (default) RAG configuration without re-indexing.

    Args:
        dataset: Evaluation samples.
        top_k: Number of chunks to retrieve.
        use_llm_judge: Override for LLM judge setting.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Evaluation results dict.
    """
    if use_llm_judge is None:
        use_llm_judge = config.evaluation.llm_judge

    logger.info(
        "Evaluating current config: model=%s, chunk=%d/overlap=%d, top_k=%d",
        config.embedding.model_name,
        config.chunk.size,
        config.chunk.overlap,
        top_k,
    )

    retriever = Retriever()
    evaluator = RAGEvaluator(top_k=top_k, use_llm_judge=use_llm_judge)
    results = evaluator.evaluate(dataset, retriever)

    if log_to_mlflow:
        run_name = (
            f"eval_{config.embedding.model_name.split('/')[-1]}"
            f"_c{config.chunk.size}_o{config.chunk.overlap}"
        )
        log_run(
            run_name=run_name,
            params={
                "embedding_model": config.embedding.model_name,
                "chunk_size": config.chunk.size,
                "chunk_overlap": config.chunk.overlap,
                "top_k": top_k,
            },
            metrics=results["global_metrics"],
            artifact_data=results,
        )

    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_experiment_summary(
    results: list[dict[str, Any]],
    group_by: str = "embedding_model",
) -> None:
    """Print a summary table of experiment results to stdout."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    header = f"{'Configuration':<35} {'Recall@k':>9} {'MRR':>9} {'Prec@k':>9} {'Index(s)':>9}"
    print(header)
    print("-" * 70)
    for r in results:
        cfg = r.get("config", {})
        if group_by == "embedding_model":
            label = cfg.get("embedding_model", "?").split("/")[-1][:34]
        else:
            label = f"c{cfg.get('chunk_size','?')}/o{cfg.get('chunk_overlap','?')}"

        gm = r.get("global_metrics", {})
        rec = gm.get("recall_at_k")
        mrr = gm.get("mrr")
        prec = gm.get("precision_at_k")
        idx_t = r.get("timing", {}).get("index_time_s")

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "  N/A "

        print(f"{label:<35} {_fmt(rec):>9} {_fmt(mrr):>9} {_fmt(prec):>9} {_fmt(idx_t):>9}")

    print("=" * 70 + "\n")
