# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Non-root user
RUN useradd -m -r raglab && chown -R raglab:raglab /app
USER raglab

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
