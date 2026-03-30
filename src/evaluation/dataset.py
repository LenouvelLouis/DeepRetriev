"""Evaluation dataset loading and validation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import config

logger = logging.getLogger(__name__)

Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class EvalSample:
    """A single evaluation question-answer pair."""

    id: int
    question: str
    expected_answer: str
    expected_sources: list[str]
    difficulty: Difficulty


def load_dataset(path: str | Path | None = None) -> list[EvalSample]:
    """Load the evaluation dataset from a JSON file.

    Args:
        path: Path to the JSON file. Defaults to config.evaluation.dataset_path.

    Returns:
        List of EvalSample objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset format is invalid.
    """
    dataset_path = Path(path or config.evaluation.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {dataset_path}. "
            "Expected a JSON file with question/answer pairs."
        )

    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must be a list of objects.")

    samples: list[EvalSample] = []
    for i, item in enumerate(raw):
        try:
            samples.append(
                EvalSample(
                    id=item["id"],
                    question=item["question"],
                    expected_answer=item["expected_answer"],
                    expected_sources=item["expected_sources"],
                    difficulty=item["difficulty"],
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Dataset item {i} is missing required field: {exc}"
            ) from exc

    logger.info("Loaded %d evaluation samples from %s", len(samples), dataset_path)
    return samples


def validate_dataset(samples: list[EvalSample]) -> bool:
    """Validate dataset integrity and report issues.

    Args:
        samples: List of EvalSample objects to validate.

    Returns:
        True if valid, False if any issues were found.
    """
    valid = True
    valid_difficulties = {"easy", "medium", "hard"}

    for sample in samples:
        if not sample.question.strip():
            logger.warning("Sample %d has an empty question.", sample.id)
            valid = False
        if not sample.expected_sources:
            logger.warning("Sample %d has no expected_sources.", sample.id)
            valid = False
        if sample.difficulty not in valid_difficulties:
            logger.warning(
                "Sample %d has invalid difficulty: %r (must be one of %s)",
                sample.id,
                sample.difficulty,
                valid_difficulties,
            )
            valid = False

    ids = [s.id for s in samples]
    if len(ids) != len(set(ids)):
        logger.warning("Dataset contains duplicate IDs.")
        valid = False

    if valid:
        logger.info("Dataset validation passed (%d samples).", len(samples))
    else:
        logger.warning("Dataset validation found issues.")

    return valid


def get_difficulty_breakdown(samples: list[EvalSample]) -> dict[str, int]:
    """Return count of samples per difficulty level.

    Args:
        samples: List of EvalSample objects.

    Returns:
        Dict mapping difficulty → count.
    """
    breakdown: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for s in samples:
        breakdown[s.difficulty] = breakdown.get(s.difficulty, 0) + 1
    return breakdown
