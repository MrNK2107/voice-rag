FROM python:3.10-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache/transformers \
    TORCH_HOME=/tmp/hf_cache/torch \
    HF_HUB_CACHE=/tmp/hf_cache/hub \
    QDRANT_PATH=/tmp/storage/qdrant_db \
    SQLITE_FTS_PATH=/tmp/storage/chunks.sqlite

RUN mkdir -p /tmp/hf_cache /tmp/storage && chown -R user:user /tmp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

CMD ["sh", "-c", "python scripts/build_index.py --languages hin --max-rows 500 && python app.py"]
