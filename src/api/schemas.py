"""Pydantic v2 request/response models for the RAGLab API."""

from pydantic import BaseModel, Field
from typing import Any


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Ask a question against the RAG knowledge base."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    """Ingest Wikipedia pages into the vector store."""

    topics: list[str] | None = None  # defaults to config.wikipedia.topics
    reset: bool = False


class EvalRequest(BaseModel):
    """Run the evaluation framework on the eval dataset."""

    top_k: int = Field(default=5, ge=1, le=50)
    llm_judge: bool = False


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    """Metadata for a single retrieved source chunk."""

    title: str
    source: str
    score: float
    text_preview: str  # first 200 chars


class QueryResponse(BaseModel):
    """Response for a RAG query."""

    answer: str
    sources: list[SourceInfo]
    confidence: float  # mean cosine similarity of top chunks
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    status: str
    documents_loaded: int
    chunks_indexed: int
    collection: str


class HealthResponse(BaseModel):
    """System health status."""

    status: str  # "healthy" or "degraded"
    api: bool
    ollama: bool
    chromadb: bool
    chromadb_chunks: int
    embedding_model: str
    llm_model: str


class MetricsResponse(BaseModel):
    """Request metrics snapshot."""

    total_requests: int
    requests_by_endpoint: dict[str, int]
    avg_latency_s: dict[str, float]
    request_counts_detail: list[dict[str, Any]]


class EvalResponse(BaseModel):
    """Evaluation framework results."""

    global_metrics: dict[str, Any]
    per_difficulty: dict[str, Any]
    n_samples: int
    config: dict[str, Any]
