FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --all-extras
COPY src ./src
RUN uv sync --frozen --no-dev --all-extras
EXPOSE 8080
CMD ["uv", "run", "--no-sync", "ragstack", "serve", "--port", "8080"]
