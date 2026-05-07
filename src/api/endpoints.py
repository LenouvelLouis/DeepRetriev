"""Route handlers for the RAGLab API."""

import asyncio
import logging
import time
from functools import partial

import requests
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from src.config import config

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and generate an answer."""
    from src.generation.generator import generate_answer
    from src.retrieval.retriever import Retriever

    loop = asyncio.get_running_loop()
    t_total_start = time.perf_counter()

    # Retrieval
    t_ret_start = time.perf_counter()
    retriever = Retriever()
    try:
        chunks = await loop.run_in_executor(
            None, partial(retriever.retrieve, req.question, top_k=req.top_k)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    retrieval_time = time.perf_counter() - t_ret_start

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. You may need to ingest documents first.",
        )

    # Generation
    t_gen_start = time.perf_counter()
    try:
        answer = await loop.run_in_executor(
            None, partial(generate_answer, req.question, chunks)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    generation_time = time.perf_counter() - t_gen_start

    total_time = time.perf_counter() - t_total_start

    # Build sources list
    sources = [
        SourceInfo(
            title=c.title,
            source=c.source,
            score=round(c.score, 4),
            text_preview=c.text[:200],
        )
        for c in chunks
    ]

    # Confidence = mean similarity score
    confidence = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 4),
        retrieval_time_s=round(retrieval_time, 4),
        generation_time_s=round(generation_time, 4),
        total_time_s=round(total_time, 4),
    )


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Load Wikipedia pages, chunk, embed, and index into ChromaDB."""
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.indexer import get_collection_stats, index_chunks
    from src.ingestion.loader import load_wikipedia_pages

    loop = asyncio.get_running_loop()

    topics = req.topics or config.wikipedia.topics

    try:
        documents = await loop.run_in_executor(
            None, partial(load_wikipedia_pages, topics=topics)
        )
    except Exception as exc:
        logger.exception("Ingestion failed during document loading")
        raise HTTPException(status_code=500, detail=f"Document loading failed: {exc}") from exc

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No documents were loaded. Check topics or network connectivity.",
        )

    try:
        chunks = await loop.run_in_executor(
            None, partial(chunk_documents, documents)
        )
        await loop.run_in_executor(
            None, partial(index_chunks, chunks, reset=req.reset)
        )
    except Exception as exc:
        logger.exception("Ingestion failed during chunking/indexing")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    stats = get_collection_stats()

    return IngestResponse(
        status="success",
        documents_loaded=len(documents),
        chunks_indexed=len(chunks),
        collection=stats["collection"],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check health of API, Ollama, and ChromaDB."""
    from src.ingestion.indexer import get_collection_stats

    # Ollama check
    ollama_ok = False
    try:
        resp = requests.get(
            config.ollama.base_url.rstrip("/") + "/api/tags",
            timeout=5,
        )
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # ChromaDB check
    chromadb_ok = False
    chromadb_chunks = 0
    try:
        stats = get_collection_stats()
        chromadb_ok = True
        chromadb_chunks = stats["count"]
    except Exception:
        pass

    status = "healthy" if (ollama_ok and chromadb_ok) else "degraded"

    return HealthResponse(
        status=status,
        api=True,
        ollama=ollama_ok,
        chromadb=chromadb_ok,
        chromadb_chunks=chromadb_chunks,
        embedding_model=config.embedding.model_name,
        llm_model=config.ollama.model,
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return request metrics collected by the middleware."""
    from src.api.metrics import get_metrics

    data = get_metrics()
    return MetricsResponse(**data)


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest) -> EvalResponse:
    """Run the RAG evaluation framework on the eval dataset."""
    from src.evaluation.dataset import load_dataset
    from src.evaluation.evaluator import RAGEvaluator
    from src.retrieval.retriever import Retriever

    loop = asyncio.get_running_loop()

    try:
        dataset = load_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    retriever = Retriever()
    evaluator = RAGEvaluator(top_k=req.top_k, use_llm_judge=req.llm_judge)

    try:
        results = await loop.run_in_executor(
            None, partial(evaluator.evaluate, dataset, retriever)
        )
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return EvalResponse(
        global_metrics=results["global_metrics"],
        per_difficulty=results["per_difficulty"],
        n_samples=results["n_samples"],
        config=results["config"],
    )
