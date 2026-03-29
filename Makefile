.PHONY: install ingest query test clean

VENV     := .venv
PYTHON   := $(VENV)/Scripts/python
PIP      := $(VENV)/Scripts/pip

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✓ Virtual environment ready. Activate with:"
	@echo "  Windows (cmd) : $(VENV)\\Scripts\\activate"
	@echo "  Windows (bash): source $(VENV)/Scripts/activate"

# ── Pipeline ─────────────────────────────────────────────────────────────────

ingest:
	$(PYTHON) main.py ingest

ingest-reset:
	$(PYTHON) main.py ingest --reset

query:
ifndef q
	$(error Usage: make query q="your question here")
endif
	$(PYTHON) main.py query "$(q)"

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) data/chroma __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean."
