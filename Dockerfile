FROM python:3.11-slim

WORKDIR /app

# torch/faiss/sentence-transformers pull in real build deps for some
# platforms -- keep it simple/robust rather than chase a minimal image.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model + index persist outside the container image so a redeploy doesn't
# require re-downloading the model or re-indexing every repo.
VOLUME ["/data"]
ENV OBSERVE_INDEX_DIR=/data/observe-index

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
