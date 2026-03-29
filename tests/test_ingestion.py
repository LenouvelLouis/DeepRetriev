"""Unit tests for ingestion components."""

import pytest

from src.ingestion.loader import Document
from src.ingestion.chunker import Chunk, chunk_document, chunk_documents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_doc() -> Document:
    return Document(
        doc_id="test:sample",
        title="Sample Document",
        text="A" * 1000,  # 1000 characters
        source="https://example.com/sample",
    )


@pytest.fixture
def short_doc() -> Document:
    return Document(
        doc_id="test:short",
        title="Short Document",
        text="Hello world.",
        source="https://example.com/short",
    )


@pytest.fixture
def empty_doc() -> Document:
    return Document(
        doc_id="test:empty",
        title="Empty Document",
        text="   ",  # whitespace only
        source="https://example.com/empty",
    )


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_basic_chunking_produces_chunks(self, sample_doc: Document) -> None:
        chunks = chunk_document(sample_doc, chunk_size=200, overlap=20)
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self, sample_doc: Document) -> None:
        chunks = chunk_document(sample_doc, chunk_size=200, overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_within_size(self, sample_doc: Document) -> None:
        chunk_size = 200
        chunks = chunk_document(sample_doc, chunk_size=chunk_size, overlap=20)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size

    def test_chunk_index_sequential(self, sample_doc: Document) -> None:
        chunks = chunk_document(sample_doc, chunk_size=200, overlap=20)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_propagated(self, sample_doc: Document) -> None:
        chunks = chunk_document(sample_doc, chunk_size=200, overlap=20)
        for chunk in chunks:
            assert chunk.doc_id == sample_doc.doc_id
            assert chunk.title == sample_doc.title
            assert chunk.source == sample_doc.source

    def test_short_doc_produces_one_chunk(self, short_doc: Document) -> None:
        chunks = chunk_document(short_doc, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0].text == short_doc.text.strip()

    def test_empty_doc_produces_no_chunks(self, empty_doc: Document) -> None:
        chunks = chunk_document(empty_doc, chunk_size=512, overlap=64)
        assert chunks == []

    def test_overlap_creates_more_chunks_than_no_overlap(self, sample_doc: Document) -> None:
        chunks_with_overlap = chunk_document(sample_doc, chunk_size=200, overlap=100)
        chunks_no_overlap = chunk_document(sample_doc, chunk_size=200, overlap=0)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_invalid_chunk_size_raises(self, sample_doc: Document) -> None:
        with pytest.raises(ValueError):
            chunk_document(sample_doc, chunk_size=0, overlap=0)

    def test_invalid_overlap_raises(self, sample_doc: Document) -> None:
        with pytest.raises(ValueError):
            chunk_document(sample_doc, chunk_size=100, overlap=100)  # overlap >= chunk_size

    def test_negative_overlap_raises(self, sample_doc: Document) -> None:
        with pytest.raises(ValueError):
            chunk_document(sample_doc, chunk_size=100, overlap=-1)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_returns_flat_list(self, sample_doc: Document, short_doc: Document) -> None:
        chunks = chunk_documents([sample_doc, short_doc], chunk_size=200, overlap=20)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_input_returns_empty(self) -> None:
        chunks = chunk_documents([])
        assert chunks == []

    def test_all_doc_ids_present(self, sample_doc: Document, short_doc: Document) -> None:
        chunks = chunk_documents([sample_doc, short_doc], chunk_size=200, overlap=20)
        doc_ids = {c.doc_id for c in chunks}
        assert sample_doc.doc_id in doc_ids
        assert short_doc.doc_id in doc_ids
