"""Cross-encoder re-ranking for retrieved chunks."""

import logging

from sentence_transformers import CrossEncoder

from src.config import config
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Re-rank retrieved chunks using a cross-encoder model.

    The cross-encoder is loaded lazily on first use and cached for subsequent calls.
    """

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading reranker model: %s", config.retrieval.reranker_model)
            self._model = CrossEncoder(config.retrieval.reranker_model)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-rank chunks by cross-encoder relevance score.

        Args:
            query: The user query.
            chunks: Chunks returned by the initial retriever.
            top_k: Number of chunks to keep after re-ranking.
                   Defaults to config.retrieval.reranker_top_k.

        Returns:
            Re-ranked list of RetrievedChunk, truncated to top_k.
        """
        if not chunks:
            return chunks

        model = self._get_model()
        pairs = [[query, c.text] for c in chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        top_k = top_k or config.retrieval.reranker_top_k

        logger.info("Re-ranked %d chunks, returning top %d", len(chunks), top_k)
        return reranked[:top_k]
