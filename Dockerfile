# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

# ==== Build stage ====
FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir hatch
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN hatch build -t wheel

# ==== CPU runtime ====
FROM python:${PYTHON_VERSION}-slim AS cpu
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY --from=build /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && pip install --no-cache-dir "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/*.whl
RUN useradd -m -u 1000 chatterbox
USER chatterbox
EXPOSE 10200
VOLUME ["/models", "/voices"]
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import socket; s=socket.create_connection(('localhost',10200),2); s.close()"
CMD ["wyoming-chatterbox"]

# ==== CUDA runtime ====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS cuda
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && pip install --no-cache-dir "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121 \
    && rm /tmp/*.whl
RUN useradd -m -u 1000 chatterbox
USER chatterbox
EXPOSE 10200
VOLUME ["/models", "/voices"]
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import socket; s=socket.create_connection(('localhost',10200),2); s.close()"
CMD ["wyoming-chatterbox"]
