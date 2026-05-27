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
    SENTIMENT_MODEL=distilbert/distilbert-base-uncased-finetuned-sst-2-english

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir -p /app/hf-cache static/uploads models \
    && python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${SENTIMENT_MODEL}')"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
