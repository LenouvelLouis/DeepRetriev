# RAGLab

A minimal, from-scratch **Retrieval-Augmented Generation** pipeline built for
learning and portfolio purposes. Every component is explicit and replaceable —
no LangChain, no magic.

Corpus: 10 Wikipedia articles on renewable energy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION                               │
│                                                                 │
│  Wikipedia / .txt   ──►  Chunker  ──►  Embedder  ──►  ChromaDB │
│  (loader.py)            (chunker.py)  (sentence-  (indexer.py) │
│                                        transformers)            │
└─────────────────────────────────────────────────────────────────┘
                                                    │
                                                    │ persist
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                          QUERY                                  │
│                                                                 │
│  User question  ──►  Retriever  ──►  Prompt builder  ──►  Ollama│
│                    (retriever.py)    (prompt.py)    (generator) │
│                                                                 │
│                                          Answer + Sources       │
└─────────────────────────────────────────────────────────────────┘
                                                    │
                                                    │ Phase 2
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                │
│                                                                 │
│  Eval dataset  ──►  Retrieval metrics  ──►  LLM-as-judge       │
│  (45 Q/A pairs)    (Recall@k, MRR,       (Faithfulness,       │
│                     Precision@k)          Relevancy, Ctx Prec) │
│                                                                 │
│  Experiments: 3 embedding models × 6 chunking configs          │
│  Tracking: MLflow    Notebook: experiments.ipynb                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Ollama | latest — https://ollama.ai |
| Mistral model | `ollama pull mistral` |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/LenouvelLouis/DeepRetriev.git
cd DeepRetriev

# 2. Create venv + install dependencies
make install

# 3. Activate the virtual environment
# Windows (bash / Git Bash):
source .venv/Scripts/activate
# Windows (cmd):
.venv\Scripts\activate
```

---

## Quick Start

```bash
# Step 1 — ingest Wikipedia articles into ChromaDB
make ingest

# Step 2 — ask a question
make query q="What is the most widely used renewable energy source?"

# Force re-ingest (wipes and rebuilds the collection)
make ingest-reset

# Run unit tests
make test
```

Or use `python` directly:

```bash
python main.py ingest
python main.py ingest --reset

python main.py query "How does solar photovoltaic technology work?"
python main.py query "What are the environmental impacts of wind farms?" --top-k 8
```

---

## Phase 2 — Experiments

Run the experiment suite to compare embedding models and chunking configurations:

```bash
# Compare 3 embedding models
python run_experiments.py --experiment embeddings

# Compare 6 chunking configurations
python run_experiments.py --experiment chunking

# Run all experiments
python run_experiments.py --experiment all

# Evaluate current config only (no re-indexing)
python run_experiments.py --evaluate

# Enable LLM-as-judge generation metrics (requires Ollama running)
python run_experiments.py --experiment all --llm-judge

# View results in MLflow
mlflow ui
```

Then open `notebooks/experiments.ipynb` for visualisations and analysis.

---

## Configuration

All parameters live in `src/config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `chunk.size` | 512 | Characters per chunk |
| `chunk.overlap` | 64 | Overlap between consecutive chunks |
| `embedding.model_name` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `chroma.collection_name` | `raglab` | ChromaDB collection |
| `ollama.model` | `mistral` | LLM served by Ollama |
| `ollama.temperature` | 0.2 | Generation temperature |
| `retrieval.top_k` | 5 | Chunks retrieved per query |
| `wikipedia.topics` | 10 renewable energy pages | Articles to ingest |

---

## Tech Stack

| Component | Library |
|---|---|
| Wikipedia loading | `wikipedia-api` |
| Chunking | custom character splitter |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `chromadb` (persistent, local) |
| LLM | Ollama (`mistral` / `llama3`) via REST API |
| Evaluation | custom metrics + LLM-as-judge |
| Experiment tracking | MLflow |
| Visualisation | matplotlib, seaborn, pandas |
| CLI | `argparse` |
| Tests | `pytest` |

---

## Project Structure

```
raglab/
├── src/
│   ├── config.py            # centralised configuration
│   ├── ingestion/
│   │   ├── loader.py        # Wikipedia + file loading
│   │   ├── chunker.py       # character-level chunking
│   │   └── indexer.py       # embedding + ChromaDB upsert
│   ├── retrieval/
│   │   └── retriever.py     # semantic search
│   ├── generation/
│   │   ├── prompt.py        # RAG prompt template
│   │   └── generator.py     # Ollama REST call
│   └── evaluation/
│       ├── dataset.py       # eval dataset loading & validation
│       ├── evaluator.py     # retrieval & generation metrics
│       ├── experiments.py   # embedding & chunking experiments
│       └── tracking.py      # MLflow wrapper
├── data/
│   └── eval_dataset.json    # 45 Q/A evaluation pairs
├── notebooks/
│   └── experiments.ipynb    # analysis & visualisations
├── tests/
│   └── test_ingestion.py
├── run_experiments.py       # experiment orchestrator CLI
├── main.py                  # CLI entry point
├── requirements.txt
├── Makefile
└── .gitignore
```

---

## Roadmap

- [x] **Phase 1** — Foundations: ingestion, chunking, embedding, retrieval, generation CLI.
- [x] **Phase 2** — Data Science: evaluation dataset, metrics framework, embedding & chunking experiments, MLflow tracking, analysis notebook.
- [ ] **Phase 3** — Serving: FastAPI, Docker, monitoring, CI/CD.
- [ ] **Phase 4** — Bonus: hybrid search, re-ranker, Streamlit UI, incremental ingestion.
