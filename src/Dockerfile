# Production Dockerfile for ADRS API
# Multi-stage build to reduce memory pressure during dependency installation

# Stage 1: Build dependencies
FROM quay.io/fedora/python-312:latest AS builder

WORKDIR /app
USER 0

# Install build dependencies
RUN dnf install -y postgresql-devel gcc && dnf clean all

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with minimal memory usage
ENV UV_NO_CACHE=1
RUN uv sync --frozen --no-dev

# Stage 2: Runtime image
FROM quay.io/fedora/python-312:latest

WORKDIR /app
USER 0

# Install only runtime dependencies (no gcc needed)
RUN dnf install -y libpq && dnf clean all

# Copy uv and virtual environment from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY pyproject.toml uv.lock ./
COPY *.py ./
COPY analytics/ ./analytics/
COPY static/ ./static/

# Create data directory for uploaded campaigns
RUN mkdir -p /app/data

# Red Hat UBI images run as user 1001 by default
# Ensure proper ownership for OpenShift
RUN chown -R 1001:0 /app && chmod -R g=u /app
USER 1001

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import httpx; httpx.get('http://localhost:8000/health')"

# Run the API server using uv run
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
