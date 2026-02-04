FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl ffmpeg software-properties-common && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone LTX-2
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app
WORKDIR /app

# Install dependencies
RUN uv sync --frozen

# Create model download script
RUN mkdir -p /models
COPY download_models.sh /app/download_models.sh
RUN chmod +x /app/download_models.sh

# Copy our API server
COPY server.py /app/server.py

# Expose API port (internal only)
EXPOSE 8090

# Download models on first run, then start server
CMD ["/bin/bash", "-c", "/app/download_models.sh && /app/.venv/bin/python3 /app/server.py"]
