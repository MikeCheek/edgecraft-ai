FROM python:3.11-slim

WORKDIR /app

# ffmpeg: decode compressed audio (e.g. browser WebM recordings)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir -r requirements-optional.txt || echo "tflite-micro not available for this platform")

COPY backend/app ./app

ENV EDGECRAFT_STORAGE_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
