FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    FLASK_DEBUG=0 \
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SENTIMENT_MODEL=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
    GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=180

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir -p /app/hf-cache static/uploads models \
    && python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')"

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --workers ${GUNICORN_WORKERS} --threads 4 --timeout ${GUNICORN_TIMEOUT} --preload --bind 0.0.0.0:${PORT} app:app"]
