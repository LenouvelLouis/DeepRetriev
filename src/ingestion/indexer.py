"""Embedding generation and ChromaDB indexing."""

import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import config
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def _get_chroma_client() -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=config.chroma.persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_create_collection(
    client: chromadb.ClientAPI,
) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    return client.get_or_create_collection(
        name=config.chroma.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], reset: bool = False) -> None:
    """Embed chunks and store them in ChromaDB.

    Args:
        chunks: List of Chunk objects to index.
        reset: If True, delete and recreate the collection before indexing.
    """
    if not chunks:
        logger.warning("No chunks to index.")
        return

    logger.info("Loading embedding model: %s", config.embedding.model_name)
    model = SentenceTransformer(config.embedding.model_name)

    client = _get_chroma_client()

    if reset:
        logger.info("Resetting collection '%s'", config.chroma.collection_name)
        try:
            client.delete_collection(config.chroma.collection_name)
        except Exception:
            pass

    collection = _get_or_create_collection(client)

    # Build batch data
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "title": c.title,
            "source": c.source,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    logger.info(
        "Embedding %d chunks with batch size %d ...",
        len(chunks),
        config.embedding.batch_size,
    )
    embeddings = model.encode(
        texts,
        batch_size=config.embedding.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    # Upsert in batches of 500 to avoid Chroma limits
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.upsert(
            ids=ids[batch_slice],
            documents=texts[batch_slice],
            embeddings=embeddings[batch_slice],
            metadatas=metadatas[batch_slice],
        )
        logger.debug("Upserted batch %d–%d", i, min(i + batch_size, len(ids)))

    logger.info(
        "Indexed %d chunks into collection '%s'",
        len(chunks),
        config.chroma.collection_name,
    )


def get_collection_stats() -> dict:
    """Return basic stats about the current ChromaDB collection.

    Returns:
        Dict with 'collection' name and 'count' of stored chunks.
    """
    client = _get_chroma_client()
    collection = _get_or_create_collection(client)
    return {
        "collection": config.chroma.collection_name,
        "count": collection.count(),
    }
