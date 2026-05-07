"""Unit tests for the evaluation module (dataset loading and retrieval metrics).

All tests run offline — no Ollama, no ChromaDB, no network calls.
"""

import pytest
from pathlib import Path

from src.evaluation.dataset import (
    EvalSample,
    load_dataset,
    validate_dataset,
    get_difficulty_breakdown,
)
from src.evaluation.evaluator import (
    _source_matches,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
)
from src.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"


@pytest.fixture
def dataset() -> list[EvalSample]:
    """Load the evaluation dataset from disk."""
    return load_dataset(DATASET_PATH)


def _make_chunk(title: str, index: int = 0, score: float = 0.9) -> RetrievedChunk:
    """Helper to build a RetrievedChunk with minimal boilerplate."""
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        title=title,
        text=f"Text for {title}",
        source=f"https://example.com/{index}",
        chunk_index=index,
        score=score,
    )


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_dataset_returns_samples(self, dataset: list[EvalSample]) -> None:
        assert len(dataset) == 45

    def test_validate_dataset_passes(self, dataset: list[EvalSample]) -> None:
        assert validate_dataset(dataset) is True

    def test_difficulty_breakdown(self, dataset: list[EvalSample]) -> None:
        breakdown = get_difficulty_breakdown(dataset)
        assert breakdown["easy"] == 11
        assert breakdown["medium"] == 20
        assert breakdown["hard"] == 14


# ---------------------------------------------------------------------------
# Source matching
# ---------------------------------------------------------------------------

class TestSourceMatches:
    def test_source_matches_case_insensitive(self) -> None:
        assert _source_matches("Solar Energy", ["solar energy"]) is True
        assert _source_matches("solar energy", ["SOLAR ENERGY"]) is True

    def test_source_matches_partial(self) -> None:
        # chunk title is substring of expected source (or vice-versa)
        assert _source_matches("Biomass", ["Biomass energy"]) is True
        assert _source_matches("Biomass energy", ["Biomass"]) is True

    def test_source_matches_no_match(self) -> None:
        assert _source_matches("Quantum computing", ["Solar energy"]) is False


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_recall_at_k_perfect(self) -> None:
        expected = ["Solar energy", "Wind power"]
        chunks = [_make_chunk("Solar energy", 0), _make_chunk("Wind power", 1)]
        assert recall_at_k(chunks, expected, k=5) == 1.0

    def test_recall_at_k_none(self) -> None:
        expected = ["Solar energy", "Wind power"]
        chunks = [_make_chunk("Quantum computing", 0), _make_chunk("Dark matter", 1)]
        assert recall_at_k(chunks, expected, k=5) == 0.0


class TestPrecisionAtK:
    def test_precision_at_k_all_relevant(self) -> None:
        expected = ["Solar energy", "Wind power"]
        chunks = [_make_chunk("Solar energy", 0), _make_chunk("Wind power", 1)]
        assert precision_at_k(chunks, expected, k=2) == 1.0

    def test_precision_at_k_none_relevant(self) -> None:
        expected = ["Solar energy"]
        chunks = [_make_chunk("Quantum computing", 0), _make_chunk("Dark matter", 1)]
        assert precision_at_k(chunks, expected, k=2) == 0.0


class TestMeanReciprocalRank:
    def test_mrr_first_hit(self) -> None:
        expected = ["Solar energy"]
        chunks = [_make_chunk("Solar energy", 0), _make_chunk("Wind power", 1)]
        assert mean_reciprocal_rank(chunks, expected) == 1.0

    def test_mrr_no_hit(self) -> None:
        expected = ["Geothermal energy"]
        chunks = [_make_chunk("Quantum computing", 0), _make_chunk("Dark matter", 1)]
        assert mean_reciprocal_rank(chunks, expected) == 0.0
