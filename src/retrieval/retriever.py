"""Semantic and hybrid retrieval from ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by a retrieval query."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity distance (lower = more similar for L2; 1-distance for cosine)


class Retriever:
    """Semantic retriever backed by ChromaDB and sentence-transformers.

    The embedding model is loaded once on first use and reused across queries.
    Supports hybrid search (BM25 + cosine) with Reciprocal Rank Fusion.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None
        # BM25 index cache (built lazily on first hybrid query)
        self._bm25_index: BM25Okapi | None = None
        self._bm25_docs: list[dict[str, Any]] | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", config.embedding.model_name)
            self._model = SentenceTransformer(config.embedding.model_name)
        return self._model

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = chromadb.PersistentClient(
                path=config.chroma.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=config.chroma.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _build_bm25_index(self) -> None:
        """Fetch all documents from ChromaDB and build a BM25 index."""
        collection = self._get_collection()
        all_data = collection.get(include=["documents", "metadatas"])

        ids: list[str] = all_data["ids"]
        documents: list[str] = all_data["documents"]
        metadatas: list[dict[str, Any]] = all_data["metadatas"]

        # Store document info for later retrieval
        self._bm25_docs = []
        tokenized_corpus: list[list[str]] = []

        for i, chunk_id in enumerate(ids):
            text = documents[i]
            meta = metadatas[i]
            self._bm25_docs.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
            })
            tokenized_corpus.append(text.lower().split())

        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents", len(ids))

    def _get_bm25_index(self) -> tuple[BM25Okapi, list[dict[str, Any]]]:
        """Return cached BM25 index, building it on first call."""
        if self._bm25_index is None or self._bm25_docs is None:
            self._build_bm25_index()
        assert self._bm25_index is not None
        assert self._bm25_docs is not None
        return self._bm25_index, self._bm25_docs

    def invalidate_bm25_cache(self) -> None:
        """Force rebuild of BM25 index on next hybrid query."""
        self._bm25_index = None
        self._bm25_docs = None

    def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Pure semantic retrieval via ChromaDB cosine similarity."""
        collection = self._get_collection()
        model = self._get_model()
        query_embedding = model.encode([query], convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            if similarity < config.retrieval.score_threshold:
                continue

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    text=results["documents"][0][i],
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=similarity,
                )
            )

        return chunks

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval combining semantic search and BM25 with RRF fusion.

        1. Runs semantic search via ChromaDB (cosine similarity).
        2. Runs BM25 keyword search over all indexed chunks.
        3. Fuses both ranked lists using Reciprocal Rank Fusion (RRF):
           ``score(d) = sum(1 / (k + rank_i(d)))`` with k = config.retrieval.rrf_k.

        Args:
            query: Natural-language question or search string.
            top_k: Number of results to return. Defaults to config.retrieval.top_k.

        Returns:
            List of RetrievedChunk ordered by fused RRF score (best first).
        """
        top_k = top_k or config.retrieval.top_k
        collection = self._get_collection()

        if collection.count() == 0:
            raise RuntimeError(
                "The ChromaDB collection is empty. Run `python main.py ingest` first."
            )

        rrf_k = config.retrieval.rrf_k

        # --- Semantic ranked list ---
        # Fetch more candidates than top_k to improve fusion quality
        n_candidates = min(top_k * 3, collection.count())
        semantic_chunks = self._retrieve_semantic(query, top_k=n_candidates)

        # chunk_id -> rank (1-based)
        semantic_ranks: dict[str, int] = {}
        semantic_data: dict[str, RetrievedChunk] = {}
        for rank, chunk in enumerate(semantic_chunks, start=1):
            semantic_ranks[chunk.chunk_id] = rank
            semantic_data[chunk.chunk_id] = chunk

        # --- BM25 ranked list ---
        bm25_index, bm25_docs = self._get_bm25_index()
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)

        # Sort by BM25 score descending, take top n_candidates
        scored_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )[:n_candidates]

        bm25_ranks: dict[str, int] = {}
        bm25_data: dict[str, dict[str, Any]] = {}
        for rank, idx in enumerate(scored_indices, start=1):
            doc = bm25_docs[idx]
            chunk_id = doc["chunk_id"]
            bm25_ranks[chunk_id] = rank
            bm25_data[chunk_id] = doc

        # --- RRF fusion ---
        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        fused_scores: dict[str, float] = {}

        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in semantic_ranks:
                score += (1.0 - config.retrieval.bm25_weight) * (
                    1.0 / (rrf_k + semantic_ranks[chunk_id])
                )
            if chunk_id in bm25_ranks:
                score += config.retrieval.bm25_weight * (
                    1.0 / (rrf_k + bm25_ranks[chunk_id])
                )
            fused_scores[chunk_id] = score

        # Sort by fused score descending
        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        # Build result chunks
        chunks: list[RetrievedChunk] = []
        for chunk_id in ranked_ids:
            fused_score = fused_scores[chunk_id]

            if chunk_id in semantic_data:
                base = semantic_data[chunk_id]
                chunks.append(
                    RetrievedChunk(
                        chunk_id=base.chunk_id,
                        doc_id=base.doc_id,
                        title=base.title,
                        text=base.text,
                        source=base.source,
                        chunk_index=base.chunk_index,
                        score=fused_score,
                    )
                )
            elif chunk_id in bm25_data:
                doc = bm25_data[chunk_id]
                meta = doc["metadata"]
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", ""),
                        title=meta.get("title", ""),
                        text=doc["text"],
                        source=meta.get("source", ""),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        score=fused_score,
                    )
                )

        logger.info(
            "Hybrid retrieval returned %d chunks for query: %r (semantic=%d, bm25=%d candidates)",
            len(chunks),
            query[:80],
            len(semantic_chunks),
            len(scored_indices),
        )
        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Delegates to hybrid retrieval when config.retrieval.use_hybrid is True,
        otherwise performs pure semantic search.

        Args:
            query: Natural-language question or search string.
            top_k: Number of results to return. Defaults to config.retrieval.top_k.

        Returns:
            List of RetrievedChunk ordered by relevance (best first).

        Raises:
            RuntimeError: If the collection is empty.
        """
        if config.retrieval.use_hybrid:
            return self.retrieve_hybrid(query, top_k=top_k)

        top_k = top_k or config.retrieval.top_k
        collection = self._get_collection()

        if collection.count() == 0:
            raise RuntimeError(
                "The ChromaDB collection is empty. Run `python main.py ingest` first."
            )

        chunks = self._retrieve_semantic(query, top_k=top_k)
        logger.info("Retrieved %d chunks for query: %r", len(chunks), query[:80])
        return chunks
