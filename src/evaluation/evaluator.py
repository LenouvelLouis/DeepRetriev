"""RAG evaluation framework — retrieval and generation metrics."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.config import config
from src.evaluation.dataset import EvalSample
from src.generation.generator import generate_answer
from src.retrieval.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-matching helpers
# ---------------------------------------------------------------------------

def _source_matches(chunk_title: str, expected_sources: list[str]) -> bool:
    """Check if a chunk title matches any expected source.

    Uses case-insensitive partial matching to handle Wikipedia redirect
    variations (e.g. 'Biomass' matching 'Biomass energy').
    """
    t = chunk_title.lower().strip()
    for source in expected_sources:
        s = source.lower().strip()
        if s in t or t in s:
            return True
    return False


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_chunks: list[RetrievedChunk],
    expected_sources: list[str],
    k: int,
) -> float:
    """Fraction of expected sources found in the top-k retrieved chunks.

    Args:
        retrieved_chunks: Ranked list of retrieved chunks (best first).
        expected_sources: Ground-truth source titles.
        k: Number of top chunks to consider.

    Returns:
        Float in [0, 1].
    """
    if not expected_sources:
        return 0.0
    top_k = retrieved_chunks[:k]
    retrieved_titles = {c.title for c in top_k}
    hits = sum(
        any(_source_matches(t, [src]) for t in retrieved_titles)
        for src in expected_sources
    )
    return hits / len(expected_sources)


def precision_at_k(
    retrieved_chunks: list[RetrievedChunk],
    expected_sources: list[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved chunks that are relevant.

    Args:
        retrieved_chunks: Ranked list of retrieved chunks.
        expected_sources: Ground-truth source titles.
        k: Number of top chunks to consider.

    Returns:
        Float in [0, 1].
    """
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if _source_matches(c.title, expected_sources))
    return relevant / len(top_k)


def mean_reciprocal_rank(
    retrieved_chunks: list[RetrievedChunk],
    expected_sources: list[str],
) -> float:
    """Reciprocal rank of the first relevant chunk.

    Args:
        retrieved_chunks: Ranked list of retrieved chunks.
        expected_sources: Ground-truth source titles.

    Returns:
        Float in (0, 1] — 0 if no relevant chunk found.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if _source_matches(chunk.title, expected_sources):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict evaluator assessing the quality of a RAG (Retrieval-Augmented "
    "Generation) system. You must provide a score from 1 to 5 and a brief justification. "
    "Always respond in this exact format:\nScore: <integer 1-5>\nJustification: <one sentence>"
)


def _call_ollama_judge(prompt: str) -> str:
    """Call Ollama with a judge prompt and return the response text."""
    try:
        response = requests.post(
            f"{config.ollama.base_url}/api/generate",
            json={
                "model": config.ollama.model,
                "prompt": f"{_JUDGE_SYSTEM}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 2048},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("Ollama judge call failed: %s", exc)
        return ""


def _parse_judge_score(response: str) -> dict[str, Any]:
    """Extract score and justification from judge response."""
    score_match = re.search(r"Score:\s*([1-5])", response, re.IGNORECASE)
    just_match = re.search(r"Justification:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    justification = just_match.group(1).strip() if just_match else response[:200]
    return {"score": score, "justification": justification}


def judge_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    """Score faithfulness: is the answer grounded in the retrieved context?

    Args:
        question: The original question.
        context: Retrieved chunks concatenated as context.
        answer: The generated answer.

    Returns:
        Dict with 'score' (1-5 or None) and 'justification'.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Context (retrieved passages):\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Does the Answer only state facts that are supported by the Context? "
        "Score 1 if the answer makes many claims not in the context (hallucination). "
        "Score 5 if every claim in the answer is directly supported by the context."
    )
    raw = _call_ollama_judge(prompt)
    return _parse_judge_score(raw)


def judge_answer_relevancy(
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Score answer relevancy: does the answer address the question?

    Args:
        question: The original question.
        answer: The generated answer.

    Returns:
        Dict with 'score' (1-5 or None) and 'justification'.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Does the Answer directly and completely address the Question? "
        "Score 1 if the answer is completely off-topic or refuses to answer. "
        "Score 5 if the answer is perfectly targeted to what was asked."
    )
    raw = _call_ollama_judge(prompt)
    return _parse_judge_score(raw)


def judge_context_precision(
    question: str,
    context: str,
) -> dict[str, Any]:
    """Score context precision: are retrieved chunks relevant to the question?

    Args:
        question: The original question.
        context: Retrieved chunks concatenated as context.

    Returns:
        Dict with 'score' (1-5 or None) and 'justification'.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Is the Retrieved Context useful for answering the Question? "
        "Score 1 if none of the context is relevant. "
        "Score 5 if all context passages are highly relevant to the question."
    )
    raw = _call_ollama_judge(prompt)
    return _parse_judge_score(raw)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context string for the judge."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c.title}\n{c.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """Evaluate a RAG pipeline on an evaluation dataset.

    Usage:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(dataset, retriever)
        evaluator.save_results(results, "data/eval_results/run_001.json")
    """

    def __init__(
        self,
        top_k: int | None = None,
        use_llm_judge: bool | None = None,
    ) -> None:
        self.top_k = top_k or config.evaluation.top_k
        self.use_llm_judge = (
            use_llm_judge if use_llm_judge is not None else config.evaluation.llm_judge
        )

    def evaluate(
        self,
        dataset: list[EvalSample],
        retriever: Retriever,
    ) -> dict[str, Any]:
        """Run full evaluation on the dataset.

        Args:
            dataset: List of evaluation samples.
            retriever: Configured Retriever instance.

        Returns:
            Results dict with global metrics, per-difficulty breakdown,
            and per-question details.
        """
        per_question: list[dict[str, Any]] = []

        logger.info(
            "Evaluating %d questions (top_k=%d, llm_judge=%s)",
            len(dataset),
            self.top_k,
            self.use_llm_judge,
        )

        for sample in tqdm(dataset, desc="Evaluating", unit="q"):
            result = self._evaluate_sample(sample, retriever)
            per_question.append(result)

        global_metrics = self._aggregate_metrics(per_question)
        per_difficulty = self._aggregate_by_difficulty(per_question)

        return {
            "global_metrics": global_metrics,
            "per_difficulty": per_difficulty,
            "per_question": per_question,
            "config": {
                "top_k": self.top_k,
                "embedding_model": config.embedding.model_name,
                "chunk_size": config.chunk.size,
                "chunk_overlap": config.chunk.overlap,
                "llm_judge": self.use_llm_judge,
            },
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(dataset),
        }

    def _evaluate_sample(
        self,
        sample: EvalSample,
        retriever: Retriever,
    ) -> dict[str, Any]:
        """Evaluate a single question."""
        t0 = time.perf_counter()

        # --- Retrieval ---
        try:
            chunks = retriever.retrieve(sample.question, top_k=self.top_k)
        except Exception as exc:
            logger.warning("Retrieval failed for Q%d: %s", sample.id, exc)
            chunks = []

        retrieval_time = time.perf_counter() - t0

        rec = recall_at_k(chunks, sample.expected_sources, self.top_k)
        prec = precision_at_k(chunks, sample.expected_sources, self.top_k)
        mrr = mean_reciprocal_rank(chunks, sample.expected_sources)

        retrieved_sources = list({c.title for c in chunks})

        result: dict[str, Any] = {
            "id": sample.id,
            "question": sample.question,
            "difficulty": sample.difficulty,
            "expected_sources": sample.expected_sources,
            "retrieved_sources": retrieved_sources,
            "recall_at_k": rec,
            "precision_at_k": prec,
            "mrr": mrr,
            "retrieval_time_s": round(retrieval_time, 4),
        }

        # --- Generation + LLM judge ---
        if self.use_llm_judge and chunks:
            try:
                answer = generate_answer(sample.question, chunks)
            except Exception as exc:
                logger.warning("Generation failed for Q%d: %s", sample.id, exc)
                answer = ""

            context = _format_context(chunks)
            result["answer"] = answer

            if answer:
                result["faithfulness"] = judge_faithfulness(
                    sample.question, context, answer
                )
                result["answer_relevancy"] = judge_answer_relevancy(
                    sample.question, answer
                )
                result["context_precision"] = judge_context_precision(
                    sample.question, context
                )
            else:
                result["faithfulness"] = {"score": None, "justification": "No answer generated"}
                result["answer_relevancy"] = {"score": None, "justification": "No answer generated"}
                result["context_precision"] = {"score": None, "justification": "No answer generated"}
        else:
            result["answer"] = None
            result["faithfulness"] = None
            result["answer_relevancy"] = None
            result["context_precision"] = None

        return result

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(per_question: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute mean metrics across all questions."""

        def _mean(key: str) -> float | None:
            vals = [q[key] for q in per_question if isinstance(q.get(key), (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _mean_judge(key: str) -> float | None:
            vals = [
                q[key]["score"]
                for q in per_question
                if q.get(key) and q[key].get("score") is not None
            ]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "recall_at_k": _mean("recall_at_k"),
            "precision_at_k": _mean("precision_at_k"),
            "mrr": _mean("mrr"),
            "faithfulness": _mean_judge("faithfulness"),
            "answer_relevancy": _mean_judge("answer_relevancy"),
            "context_precision": _mean_judge("context_precision"),
            "mean_retrieval_time_s": _mean("retrieval_time_s"),
        }

    @staticmethod
    def _aggregate_by_difficulty(
        per_question: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate metrics split by difficulty level."""
        groups: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        for q in per_question:
            groups.setdefault(q["difficulty"], []).append(q)

        result: dict[str, dict[str, Any]] = {}
        for difficulty, questions in groups.items():
            if not questions:
                continue

            def _mean(key: str, qs: list[dict] = questions) -> float | None:
                vals = [q[key] for q in qs if isinstance(q.get(key), (int, float))]
                return round(sum(vals) / len(vals), 4) if vals else None

            def _mean_judge(key: str, qs: list[dict] = questions) -> float | None:
                vals = [
                    q[key]["score"]
                    for q in qs
                    if q.get(key) and q[key].get("score") is not None
                ]
                return round(sum(vals) / len(vals), 4) if vals else None

            result[difficulty] = {
                "n": len(questions),
                "recall_at_k": _mean("recall_at_k"),
                "precision_at_k": _mean("precision_at_k"),
                "mrr": _mean("mrr"),
                "faithfulness": _mean_judge("faithfulness"),
                "answer_relevancy": _mean_judge("answer_relevancy"),
                "context_precision": _mean_judge("context_precision"),
            }

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(results: dict[str, Any], output_path: str | Path) -> Path:
        """Save evaluation results to a JSON file.

        Args:
            results: Results dict returned by evaluate().
            output_path: Destination file path.

        Returns:
            Path where results were saved.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation results saved to %s", out)
        return out
