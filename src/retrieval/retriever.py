"""Semantic retrieval from ChromaDB."""

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
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
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None

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

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language question or search string.
            top_k: Number of results to return. Defaults to config.retrieval.top_k.

        Returns:
            List of RetrievedChunk ordered by relevance (best first).

        Raises:
            RuntimeError: If the collection is empty.
        """
        top_k = top_k or config.retrieval.top_k
        collection = self._get_collection()

        if collection.count() == 0:
            raise RuntimeError(
                "The ChromaDB collection is empty. Run `python main.py ingest` first."
            )

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
            # For cosine space: similarity = 1 - distance
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

        logger.info("Retrieved %d chunks for query: %r", len(chunks), query[:80])
        return chunks
