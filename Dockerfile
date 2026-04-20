# syntax=docker/dockerfile:1.7

# ---- Base image
# python:3.11-slim-bookworm = Debian 12 with Python 3.11, minimal layer.
# We pin the minor version for reproducibility. slim is ~50MB vs 1GB for
# the full python image, while still including pip.
FROM python:3.11-slim-bookworm AS base

# ---- System setup
# Prevent Python from writing .pyc files and enable unbuffered output
# (important for logs: without this, prints get buffered and disappear
# if the container is killed).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build tooling only for the compile step (LightGBM wheel needs it
# on some architectures). libgomp1 is the OpenMP runtime LightGBM uses
# for multi-threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user for running the service (defence in depth)
# Running as non-root is a security best practice: if someone compromises
# the process, they do not automatically get root inside the container.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# ---- Python dependencies
# Copy only pyproject.toml first so that dependency layer is cached
# and rebuilds only when deps change (not when we edit our code).
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install ".[serve]"

# ---- Application code and artifacts
# Copy the source code, model artifacts, and preprocessor.
# .dockerignore keeps data/, notebooks/, tests/, etc. out.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Fix permissions so appuser can read everything
RUN chown -R appuser:appuser /app

USER appuser

# ---- Runtime configuration
# Expose the port for documentation purposes; actual mapping is done
# at `docker run` or compose time.
EXPOSE 8000

# Liveness probe: Docker periodically calls /health to restart the
# container if it becomes unresponsive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# ---- Startup command
# --host 0.0.0.0 is REQUIRED inside Docker. Default 127.0.0.1 would only
# accept connections from INSIDE the container, making the port mapping
# useless. With 0.0.0.0, the service listens on all interfaces.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
