"""Prompt templates for RAG generation."""

from src.retrieval.retriever import RetrievedChunk

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise and helpful assistant. Answer the user's \
question using ONLY the information provided in the context below. If the \
context does not contain enough information to answer the question, say so \
clearly — do not hallucinate or use outside knowledge.

When you cite a fact, you may mention the source title in parentheses."""

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------
_RAG_TEMPLATE = """{system}

### Context
{context}

### Question
{question}

### Answer"""


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the full RAG prompt from a question and retrieved chunks.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks, ordered by relevance.

    Returns:
        Formatted prompt string ready to be sent to the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] (Source: {chunk.title})\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return _RAG_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )
