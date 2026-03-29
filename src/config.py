"""Centralized configuration for RAGLab."""

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
    persist_directory: str = "./data/chroma"


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
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
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    wikipedia: WikipediaConfig = field(default_factory=WikipediaConfig)


# Singleton — import this everywhere
config = Config()
