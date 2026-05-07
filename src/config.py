"""Centralized configuration for RAGLab."""

import os
from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    size: int = 512
    overlap: int = 64


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class ChromaConfig:
    collection_name: str = "raglab"
    persist_directory: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))


@dataclass
class OllamaConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral"))
    temperature: float = 0.2
    num_ctx: int = 4096


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.0


@dataclass
class WikipediaConfig:
    language: str = "en"
    topics: list[str] = field(default_factory=lambda: [
        "Solar energy",
        "Wind power",
        "Hydropower",
        "Geothermal energy",
        "Biomass energy",
        "Tidal power",
        "Wave power",
        "Renewable energy",
        "Energy storage",
        "Photovoltaic system",
    ])


@dataclass
class EvaluationConfig:
    dataset_path: str = "./data/eval_dataset.json"
    results_dir: str = "./data/eval_results"
    top_k: int = 5
    llm_judge: bool = True
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    experiment_name: str = "raglab_experiments"
    embedding_models: list[str] = field(default_factory=lambda: [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
    ])
    chunking_configs: list[dict] = field(default_factory=lambda: [
        {"chunk_size": 256, "chunk_overlap": 0},
        {"chunk_size": 256, "chunk_overlap": 50},
        {"chunk_size": 512, "chunk_overlap": 64},
        {"chunk_size": 512, "chunk_overlap": 100},
        {"chunk_size": 1024, "chunk_overlap": 100},
        {"chunk_size": 1024, "chunk_overlap": 200},
    ])


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    wikipedia: WikipediaConfig = field(default_factory=WikipediaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Singleton — import this everywhere
config = Config()
