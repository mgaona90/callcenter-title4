FROM python:3.12-slim

WORKDIR /app

# System deps for fastembed model download and PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed models at build time so first request is instant
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
    SparseTextEmbedding(model_name='Qdrant/bm42-all-minilm-l6-v2-attentions')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
