# ============================================================
# StockPredictionPro - Dockerfile (Multi-stage, Python 3.11)
# Place this file at the project root as: Dockerfile
# ============================================================

# ---------- Base versions ----------
ARG PYTHON_VERSION=3.11-slim

# ---------- Builder stage ----------
FROM python:${PYTHON_VERSION} AS builder

# Ensure non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for scientific Python and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first (better layer caching)
COPY requirements.txt ./

# Upgrade pip and build wheels into a local wheelhouse for faster installs
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ---------- Runtime stage ----------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install minimal system packages (curl for healthchecks, fonts for charts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels /wheels/*

# Copy project files
# (Adjust if you want to exclude large folders like notebooks or dist)
COPY . /app

# Create required directories at runtime if not present
RUN mkdir -p data/raw data/processed data/models data/predictions data/backtests data/exports \
    && mkdir -p logs/audit/runs logs/audit/models \
    && chown -R appuser:appuser /app

USER appuser

# Streamlit configuration (headless, wide mode)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=true \
    STREAMLIT_THEME_BASE="dark"

# Expose ports
# 8501 -> Streamlit UI
# 8000 -> Optional FastAPI (if you choose to run it)
EXPOSE 8501 8000

# Healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit app
# To run FastAPI instead, override CMD at runtime:
#   docker run ... sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
