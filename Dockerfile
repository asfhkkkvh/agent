# ── Frontend Build Stage ──────────────────────────────────────
FROM node:22-alpine AS frontend

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# ── Python Build Stage ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for docling + sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime Stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy built frontend
COPY --from=frontend /frontend/dist /app/frontend/dist

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Non-root user for security
RUN useradd -m -u 1000 omnirag && chown -R omnirag:omnirag /app
USER omnirag

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
