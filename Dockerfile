# ---- Base image ----
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- System build dependencies ----
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies (cached separately) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Project files ----
# Copies all code files (main.py, faq_router.py, faqs.json, etc.)
COPY . .

# ---- Run as non-root user ----
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]