FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps - use Python 3.10 (ships with Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone LTX-2 repo (includes uv workspace with ltx-core + ltx-pipelines)
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app
WORKDIR /app

# Install dependencies via uv
# Use --python to ensure correct interpreter, allow network for index resolution
RUN uv sync --python python3

# Install FastAPI + uvicorn for our API server
RUN uv pip install fastapi uvicorn[standard]

# Create models directory (will be mounted as volume for persistence)
RUN mkdir -p /models /outputs

# Copy our files
COPY download_models.sh /app/download_models.sh
COPY server.py /app/server.py
RUN chmod +x /app/download_models.sh

# Expose API port (internal only)
EXPOSE 8090

# Health check - generous timeout for initial model loading
HEALTHCHECK --interval=60s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Download models on first run, then start server
CMD ["/bin/bash", "-c", "/app/download_models.sh && uv run python3 /app/server.py"]
