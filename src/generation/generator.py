"""LLM generation via the Ollama REST API."""

import json
import logging

import requests

from src.config import config
from src.generation.prompt import build_rag_prompt
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_GENERATE_ENDPOINT = "/api/generate"
_TIMEOUT_SECONDS = 120


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the generated text.

    Args:
        prompt: Full prompt string.

    Returns:
        Generated text from the model.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    url = config.ollama.base_url.rstrip("/") + _GENERATE_ENDPOINT
    payload = {
        "model": config.ollama.model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": config.ollama.temperature,
            "num_ctx": config.ollama.num_ctx,
        },
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {config.ollama.base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP error: {exc}") from exc

    # Stream and concatenate response tokens
    full_text = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        token = data.get("response", "")
        full_text.append(token)
        if data.get("done"):
            break

    return "".join(full_text).strip()


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Generate an answer for the question using retrieved chunks as context.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks.

    Returns:
        Generated answer string.

    Raises:
        RuntimeError: Propagated from _call_ollama on connection failure.
    """
    if not chunks:
        logger.warning("No context chunks provided — answer quality may be low.")

    prompt = build_rag_prompt(question, chunks)
    logger.debug("Sending prompt to Ollama (%d chars)", len(prompt))

    answer = _call_ollama(prompt)
    logger.info("Generated answer (%d chars)", len(answer))
    return answer
