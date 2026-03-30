"""MLflow experiment tracking wrapper for RAGLab."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.exceptions

from src.config import config

logger = logging.getLogger(__name__)


def init_tracking(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Initialize MLflow tracking and return the experiment name.

    Args:
        experiment_name: Name for the MLflow experiment.
                         Defaults to config.experiment.experiment_name.
        tracking_uri: MLflow tracking server URI.
                      Defaults to config.experiment.mlflow_tracking_uri.

    Returns:
        The experiment name that was set.
    """
    uri = tracking_uri or config.experiment.mlflow_tracking_uri
    name = experiment_name or config.experiment.experiment_name

    mlflow.set_tracking_uri(uri)

    try:
        mlflow.create_experiment(name)
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists

    mlflow.set_experiment(name)
    logger.info("MLflow tracking initialized: uri=%s, experiment=%s", uri, name)
    return name


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_data: dict[str, Any] | None = None,
) -> str:
    """Log a single experiment run to MLflow.

    Args:
        run_name: Human-readable name for this run.
        params: Hyperparameters / configuration values to log.
        metrics: Numeric metrics to log (None values are skipped).
        artifact_data: Optional dict that will be serialized to JSON
                       and logged as an artifact.

    Returns:
        The MLflow run_id.
    """
    init_tracking()

    # Filter out None metrics — MLflow cannot log them
    clean_metrics = {
        k: float(v) for k, v in metrics.items() if v is not None
    }
    # Flatten nested params (e.g. dicts → "key.subkey")
    flat_params = _flatten_dict(params)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flat_params)
        mlflow.log_metrics(clean_metrics)

        if artifact_data is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                json.dump(artifact_data, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="results")
            Path(tmp_path).unlink(missing_ok=True)

        run_id = run.info.run_id
        logger.info("Logged MLflow run '%s' (run_id=%s)", run_name, run_id)
        return run_id


def get_all_runs(
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve all runs for an experiment as a list of dicts.

    Useful for building summary tables outside of the MLflow UI.

    Args:
        experiment_name: Experiment to query.
                         Defaults to config.experiment.experiment_name.

    Returns:
        List of dicts with keys 'run_id', 'run_name', 'params', 'metrics'.
    """
    init_tracking(experiment_name=experiment_name)
    name = experiment_name or config.experiment.experiment_name

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        logger.warning("MLflow experiment '%s' not found.", name)
        return []

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs_df.empty:
        return []

    results: list[dict[str, Any]] = []
    for _, row in runs_df.iterrows():
        param_cols = {k[7:]: v for k, v in row.items() if k.startswith("params.")}
        metric_cols = {k[8:]: v for k, v in row.items() if k.startswith("metrics.")}
        results.append(
            {
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName", ""),
                "params": param_cols,
                "metrics": metric_cols,
            }
        )

    return results


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten a nested dict into dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
