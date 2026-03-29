"""Text chunking with configurable size and overlap."""

import logging
from dataclasses import dataclass

from src.config import config
from src.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk derived from a Document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int


def chunk_document(
    doc: Document,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split a document into overlapping text chunks.

    Uses character-level splitting. A sentence-aware splitter can be swapped
    in for Phase 2 without changing the interface.

    Args:
        doc: Source document.
        chunk_size: Max characters per chunk. Defaults to config.chunk.size.
        overlap: Overlap characters between consecutive chunks.
                 Defaults to config.chunk.overlap.

    Returns:
        List of Chunk objects ordered by position in the document.
    """
    chunk_size = chunk_size or config.chunk.size
    overlap = overlap if overlap is not None else config.chunk.overlap

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must be in [0, chunk_size), got overlap={overlap} chunk_size={chunk_size}"
        )

    text = doc.text.strip()
    if not text:
        logger.warning("Document '%s' has empty text, skipping", doc.doc_id)
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    index = 0
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk{index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    source=doc.source,
                    chunk_index=index,
                )
            )
            index += 1
        start += step

    logger.debug(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
        doc.doc_id,
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of documents.

    Args:
        documents: List of source documents.
        chunk_size: Forwarded to chunk_document.
        overlap: Forwarded to chunk_document.

    Returns:
        Flat list of all chunks across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))

    logger.info(
        "Total chunks produced: %d from %d documents", len(all_chunks), len(documents)
    )
    return all_chunks
