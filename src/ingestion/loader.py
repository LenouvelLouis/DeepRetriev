"""Document loading: Wikipedia pages and local plain-text files."""

import logging
from dataclasses import dataclass
from pathlib import Path

import wikipediaapi

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A raw document before chunking."""

    doc_id: str
    title: str
    text: str
    source: str  # URL or file path


def load_wikipedia_pages(topics: list[str] | None = None) -> list[Document]:
    """Fetch Wikipedia pages for the given topics.

    Args:
        topics: List of article titles. Defaults to config.wikipedia.topics.

    Returns:
        List of Document objects.
    """
    topics = topics or config.wikipedia.topics
    wiki = wikipediaapi.Wikipedia(
        language=config.wikipedia.language,
        user_agent="RAGLab/1.0 (portfolio project)",
    )

    documents: list[Document] = []
    for topic in topics:
        logger.info("Fetching Wikipedia page: %s", topic)
        page = wiki.page(topic)
        if not page.exists():
            logger.warning("Page not found on Wikipedia: %s", topic)
            continue
        doc = Document(
            doc_id=f"wiki:{topic.lower().replace(' ', '_')}",
            title=page.title,
            text=page.text,
            source=page.fullurl,
        )
        documents.append(doc)
        logger.debug("Loaded page '%s' (%d chars)", page.title, len(page.text))

    logger.info("Loaded %d Wikipedia documents", len(documents))
    return documents


def load_text_files(directory: str | Path) -> list[Document]:
    """Load all .txt files from a directory.

    Args:
        directory: Path to the directory containing .txt files.

    Returns:
        List of Document objects.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error("Directory does not exist: %s", directory)
        return []

    documents: list[Document] = []
    for path in sorted(directory.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
            doc = Document(
                doc_id=f"file:{path.stem}",
                title=path.stem.replace("_", " ").title(),
                text=text,
                source=str(path.resolve()),
            )
            documents.append(doc)
            logger.debug("Loaded file '%s' (%d chars)", path.name, len(text))
        except OSError as exc:
            logger.error("Failed to read file %s: %s", path, exc)

    logger.info("Loaded %d text files from %s", len(documents), directory)
    return documents
