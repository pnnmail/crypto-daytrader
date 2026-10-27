FROM python:3.11-slim

WORKDIR /app

# gcc/g++ are needed to build stumpy/scipy/numpy on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY main.py .
COPY utils/ ./utils/
COPY data/ ./data/
COPY agents/ ./agents/
COPY telegram_bot/ ./telegram_bot/

# robinhood_coins.json is copied as a DEFAULT for the first run without
# a mounted volume — with a volume mount in docker-compose.yml it is
# overridden by the real one (edits via /coins add/remove survive a rebuild).
COPY robinhood_coins.json .

# The knowledge base from the book (RAG for the decider) — 4 MB, a copy from stock_agent2.
COPY chroma_db/ ./chroma_db/

RUN mkdir -p /app/reports
RUN chmod -R 777 /app

CMD ["python", "main.py"]
