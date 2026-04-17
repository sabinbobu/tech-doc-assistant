# ── Build stage ──
# We use a two-stage build to keep the final image small.
# Stage 1: install dependencies
# Stage 2: copy only what's needed to run the app
#
# ANALOGY: Like separating your build toolchain from your production firmware.
# You don't ship the compiler with the ECU — same principle here.

FROM python:3.11-slim AS builder

# Install uv — our fast dependency manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first — Docker caches this layer.
# If only your source code changes (not dependencies),
# Docker reuses the cached install layer. Much faster rebuilds.
# Like precompiling your libraries separately from your application code.
COPY pyproject.toml .
COPY README.md .

# Install production dependencies only (no dev tools in the image)
RUN uv pip install --system --no-cache ".[eval]"


# ── Runtime stage ──
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source and install as package
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY pyproject.toml README.md ./
RUN pip install --no-deps -e .

# Create directories for runtime data
# These will be mounted as volumes in production
RUN mkdir -p data/raw data/processed vectorstore

# Streamlit config — disable telemetry and set server options
RUN mkdir -p ~/.streamlit && echo '\
[server]\n\
headless = true\n\
port = 8501\n\
[browser]\n\
gatherUsageStats = false\n\
' > ~/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check — Docker will mark container unhealthy if Streamlit stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit UI
# Override with: docker run ... python src/mcp_server/server.py
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]
