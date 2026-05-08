"""Integration tests for the RAGLab Phase 3 FastAPI application.

All tests run WITHOUT Ollama or ChromaDB by mocking external dependencies.
Uses httpx.AsyncClient with ASGITransport for in-process ASGI testing.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch

from httpx import AsyncClient, ASGITransport

from src.retrieval.retriever import RetrievedChunk

pytestmark = pytest.mark.asyncio(loop_scope="function")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a fresh FastAPI application instance."""
    from src.api.app import create_app

    return create_app()


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    """Async HTTP client wired to the ASGI app (no real server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _fake_chunks(n: int = 3) -> list[RetrievedChunk]:
    """Build a list of fake RetrievedChunk objects for mocking."""
    return [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            doc_id=f"doc-{i}",
            title=f"Fake Title {i}",
            text=f"Fake chunk text number {i}.",
            source=f"https://example.com/{i}",
            chunk_index=i,
            score=0.95 - i * 0.05,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        expected_fields = {
            "status", "api", "ollama", "chromadb",
            "chromadb_chunks", "embedding_model", "llm_model",
        }
        assert expected_fields.issubset(data.keys())

    async def test_health_api_always_true(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["api"] is True


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for GET /metrics."""

    async def test_metrics_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/metrics")
        assert response.status_code == 200

    async def test_metrics_has_expected_fields(self, client: AsyncClient) -> None:
        response = await client.get("/metrics")
        data = response.json()
        assert "total_requests" in data
        assert "requests_by_endpoint" in data
        assert "avg_latency_s" in data


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------

class TestQuery:
    """Tests for POST /query."""

    async def test_query_requires_question(self, client: AsyncClient) -> None:
        response = await client.post("/query", json={})
        assert response.status_code == 422

    async def test_query_question_too_long(self, client: AsyncClient) -> None:
        response = await client.post("/query", json={"question": "x" * 3000})
        assert response.status_code == 422

    async def test_query_invalid_top_k(self, client: AsyncClient) -> None:
        response = await client.post(
            "/query", json={"question": "What is solar energy?", "top_k": 0}
        )
        assert response.status_code == 422

    @patch("src.generation.generator.generate_answer", return_value="Mocked answer")
    @patch("src.retrieval.retriever.Retriever.retrieve")
    async def test_query_valid_request_structure(
        self,
        mock_retrieve: MagicMock,
        mock_generate: MagicMock,
        client: AsyncClient,
    ) -> None:
        mock_retrieve.return_value = _fake_chunks(3)

        response = await client.post(
            "/query", json={"question": "What is solar energy?"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "retrieval_time_s" in data
        assert "generation_time_s" in data
        assert "total_time_s" in data
        assert isinstance(data["sources"], list)


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------

class TestIngest:
    """Tests for POST /ingest."""

    @patch("src.ingestion.indexer.get_collection_stats", return_value={"collection": "raglab", "count": 5})
    @patch("src.ingestion.indexer.index_chunks")
    @patch("src.ingestion.chunker.chunk_documents", return_value=[MagicMock()])
    @patch(
        "src.ingestion.loader.load_wikipedia_pages",
        return_value=[MagicMock(doc_id="doc-1", title="Fake", text="Fake text", source="http://example.com")],
    )
    async def test_ingest_returns_expected_fields(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_index: MagicMock,
        mock_stats: MagicMock,
        client: AsyncClient,
    ) -> None:
        response = await client.post("/ingest", json={})
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "documents_loaded" in data
        assert "chunks_indexed" in data
        assert "collection" in data


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Tests for POST /evaluate."""

    async def test_evaluate_requires_valid_top_k(self, client: AsyncClient) -> None:
        response = await client.post("/evaluate", json={"top_k": 0})
        assert response.status_code == 422

    @patch("src.evaluation.evaluator.RAGEvaluator")
    @patch("src.retrieval.retriever.Retriever")
    @patch("src.evaluation.dataset.load_dataset")
    async def test_evaluate_happy_path(
        self,
        mock_load_dataset: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_evaluator_cls: MagicMock,
        client: AsyncClient,
    ) -> None:
        mock_load_dataset.return_value = [MagicMock()]
        mock_evaluator = mock_evaluator_cls.return_value
        mock_evaluator.evaluate.return_value = {
            "global_metrics": {"recall_at_k": 0.8, "mrr": 0.75},
            "per_difficulty": {"easy": {"recall_at_k": 0.9}},
            "n_samples": 10,
            "config": {"top_k": 5, "embedding_model": "all-MiniLM-L6-v2"},
        }

        response = await client.post("/evaluate", json={"top_k": 5})
        assert response.status_code == 200

        data = response.json()
        assert data["n_samples"] == 10
        assert "global_metrics" in data
        assert "per_difficulty" in data
        assert "config" in data
        assert data["global_metrics"]["recall_at_k"] == 0.8
