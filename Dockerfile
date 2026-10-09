# ---- Base image ----
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr.
# HF_HOME keeps the downloaded embedding model in a known (volume-mountable) place.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# ---- System packages ----
# Intentionally NONE. ffmpeg (and its libopus0 dependency) was the package that
# kept failing to download, and nothing in this project uses it: voice notes are
# sent to Gemini as raw bytes. build-essential/g++ are not needed because every
# dependency ships prebuilt wheels for python 3.11.
# If a future pip install ever fails with "gcc not found", re-add:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#       && rm -rf /var/lib/apt/lists/*

# ---- PyTorch (CPU-only) ----
# sentence-transformers pulls in torch; the default Linux wheel bundles ~2 GB+ of
# CUDA libraries you can't use on a 1.5 CPU container. The CPU build is far
# smaller, so the download is much less likely to time out.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# ---- Python dependencies (cached separately) ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Project files ----
# Copies all code files (main.py, faq_router.py, faqs.json, etc.)
COPY . .

# ---- Run as non-root user ----
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]