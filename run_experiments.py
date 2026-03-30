"""Phase 2 — RAGLab experiment orchestration script.

Usage:
    python run_experiments.py --experiment embeddings   # compare embedding models
    python run_experiments.py --experiment chunking     # compare chunking configs
    python run_experiments.py --experiment all          # run both experiments
    python run_experiments.py --evaluate                # evaluate current config only

Options:
    --top-k INT         Number of chunks to retrieve (default: 5)
    --llm-judge         Enable LLM-as-judge generation metrics (requires Ollama)
    --no-mlflow         Disable MLflow logging
    --results-dir PATH  Directory to save JSON results (default: ./data/eval_results)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGLab Phase 2 — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--experiment",
        choices=["embeddings", "chunking", "all"],
        help="Which experiment to run.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the current config only (no re-indexing).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-judge metrics (faithfulness, relevancy, precision). "
             "Requires Ollama to be running. Adds significant runtime.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow logging (useful for quick tests).",
    )
    parser.add_argument(
        "--results-dir",
        default="./data/eval_results",
        metavar="PATH",
        help="Directory for saving JSON results (default: ./data/eval_results).",
    )
    return parser.parse_args()


def _save_results(
    results: dict | list,
    results_dir: str,
    filename: str,
) -> Path:
    """Save results dict/list to a timestamped JSON file."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{filename}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved → %s", out_path)
    return out_path


def main() -> None:
    args = _parse_args()
    log_to_mlflow = not args.no_mlflow

    # --- Lazy imports (keep startup fast for --help) ---
    from src.config import config
    from src.evaluation.dataset import get_difficulty_breakdown, load_dataset, validate_dataset
    from src.evaluation.experiments import (
        run_chunking_experiments,
        run_embedding_experiments,
        run_evaluation_only,
    )
    from src.evaluation.tracking import init_tracking

    # --- Dataset ---
    logger.info("Loading evaluation dataset from %s", config.evaluation.dataset_path)
    try:
        dataset = load_dataset()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not validate_dataset(dataset):
        logger.warning("Dataset has validation issues — proceeding anyway.")

    breakdown = get_difficulty_breakdown(dataset)
    logger.info(
        "Dataset: %d samples (easy=%d, medium=%d, hard=%d)",
        len(dataset),
        breakdown["easy"],
        breakdown["medium"],
        breakdown["hard"],
    )

    # --- MLflow init ---
    if log_to_mlflow:
        init_tracking()
        logger.info("MLflow tracking enabled. Run `mlflow ui` to view results.")

    # ----------------------------------------------------------------
    # --evaluate: evaluate current config, no re-indexing
    # ----------------------------------------------------------------
    if args.evaluate:
        logger.info("Running evaluation-only mode (current config, no re-indexing)...")
        results = run_evaluation_only(
            dataset=dataset,
            top_k=args.top_k,
            use_llm_judge=args.llm_judge,
            log_to_mlflow=log_to_mlflow,
        )
        _save_results(results, args.results_dir, "evaluate_current")
        _print_summary(results)
        return

    # ----------------------------------------------------------------
    # --experiment embeddings / chunking / all
    # ----------------------------------------------------------------
    # Pre-load Wikipedia documents once to avoid redundant fetches
    logger.info("Loading Wikipedia documents (this may take a moment)...")
    from src.ingestion.loader import load_wikipedia_pages
    documents = load_wikipedia_pages()
    logger.info("Loaded %d documents.", len(documents))

    if args.experiment in ("embeddings", "all"):
        logger.info("=" * 50)
        logger.info("EXPERIMENT: Embedding Models")
        logger.info("=" * 50)
        emb_results = run_embedding_experiments(
            dataset=dataset,
            documents=documents,
            top_k=args.top_k,
            use_llm_judge=args.llm_judge,
        )
        _save_results(emb_results, args.results_dir, "embeddings")

    if args.experiment in ("chunking", "all"):
        logger.info("=" * 50)
        logger.info("EXPERIMENT: Chunking Configurations")
        logger.info("=" * 50)
        chunk_results = run_chunking_experiments(
            dataset=dataset,
            documents=documents,
            top_k=args.top_k,
            use_llm_judge=args.llm_judge,
        )
        _save_results(chunk_results, args.results_dir, "chunking")

    if log_to_mlflow:
        print("\n✓ Experiments complete. View results with:")
        print("    mlflow ui")
        print("  Then open http://localhost:5000 in your browser.\n")
    else:
        print("\n✓ Experiments complete. Results saved to", args.results_dir)


def _print_summary(results: dict) -> None:
    """Print a compact summary of evaluation results."""
    gm = results.get("global_metrics", {})
    cfg = results.get("config", {})
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Model       : {cfg.get('embedding_model', 'N/A')}")
    print(f"  Chunk size  : {cfg.get('chunk_size', 'N/A')} / overlap {cfg.get('chunk_overlap', 'N/A')}")
    print(f"  Top-k       : {cfg.get('top_k', 'N/A')}")
    print(f"  Samples     : {results.get('n_samples', 'N/A')}")
    print()
    print(f"  Recall@k    : {gm.get('recall_at_k') or 'N/A'}")
    print(f"  MRR         : {gm.get('mrr') or 'N/A'}")
    print(f"  Precision@k : {gm.get('precision_at_k') or 'N/A'}")
    if gm.get("faithfulness") is not None:
        print(f"  Faithfulness: {gm['faithfulness']}/5.0")
        print(f"  Relevancy   : {gm.get('answer_relevancy')}/5.0")
        print(f"  Ctx Prec.   : {gm.get('context_precision')}/5.0")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
