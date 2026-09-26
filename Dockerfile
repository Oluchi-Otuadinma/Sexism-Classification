FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first: the PyPI torch wheel bundles CUDA (~3 GB extra) that
# this CPU-only API image doesn't need. The pin matches requirements.txt, so
# the later `pip install -r` sees it as already satisfied.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies (torch already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + dataset split
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Model behaviour in the container:
#   INFERENCE_BACKEND=hf_api            -> no local weights, calls the HF Inference API
#   INFERENCE_BACKEND=local (default)   -> downloads vinai/bertweet-base to the
#       user's HF cache on startup (~500 MB, cached across restarts via a
#       mounted volume). Mount a checkpoint at /app/outputs/models to skip
#       the download entirely once one is fine-tuned.
ENV INFERENCE_BACKEND=local \
    PRELOAD_MODEL=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
