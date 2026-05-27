---
title: Fashion Trend Analysis
emoji: "\U0001F457"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Social Media Fashion Trend Analysis System

Multimodal ML pipeline for analysing fashion trends among university students in Lagos.

> The YAML block above is read by Hugging Face Spaces to configure the deployment.
> It is ignored by GitHub.

## Components

| File              | Purpose                                                 |
| ----------------- | ------------------------------------------------------- |
| `scraper.py`      | Collects social media posts (mock; swap for real API)   |
| `preprocess.py`   | Resizes images in `dataset/images/` to 224x224          |
| `image_model.py`  | CNN fashion image classifier                            |
| `text_model.py`   | BERT sentiment + rule-based trend category tagging      |
| `clustering.py`   | K-Means clustering on engagement features               |
| `forecast.py`     | LSTM trend forecasting                                  |
| `app.py`          | Flask dashboard exposing all of the above               |

## Setup

```
pip install -r requirements.txt
```

> Heavy deps (TensorFlow, PyTorch, transformers) take a while on first install.

## Quick run

```
python app.py
```

Then open http://127.0.0.1:5000

## Train / build models individually

```
python scraper.py            # builds dataset/fashion_posts.csv
python preprocess.py         # resizes dataset/images/*
python image_model.py        # saves models/cnn_model.h5  (untrained skeleton)
python clustering.py         # saves models/clustering_model.pkl
python forecast.py           # saves models/lstm_model.h5
python text_model.py         # runs three example captions
```

## Train the fashion CNN on real images

To replace the untrained CNN skeleton with a model that actually predicts
fashion categories, see [`dataset/images/README.md`](dataset/images/README.md)
for the expected folder layout, then run:

```
python train_image_model.py              # transfer learning (MobileNetV2) — recommended
python train_image_model.py --simple-cnn # plain CNN from the original spec
```

Outputs `models/cnn_model.h5`, `models/cnn_labels.json`, and `models/cnn_report.txt`
(accuracy, per-class precision/recall, confusion matrix — drop into your thesis).
The dashboard auto-detects the trained model and labels on next start.

> Training on CPU is slow; use Google Colab with a free T4 GPU if available.

## API endpoints

| Method | Endpoint              | Body                              |
| ------ | --------------------- | --------------------------------- |
| POST   | `/api/analyse-text`   | `{"text": "..."}`                 |
| POST   | `/api/analyse-image`  | multipart form, field `image`     |
| GET    | `/api/cluster`        | -                                 |
| POST   | `/api/forecast`       | `{"series": [n1, n2, ...]}`       |

## Notes vs. original spec

Fixed during implementation:

- `**name**` → `__name__`, `"**main**"` → `"__main__"`
- Restored indentation lost in the markdown-formatted spec
- Added missing imports and error handling at file/IO boundaries
- CNN dropped from a 4-class to a configurable n-class model with a second conv block
- Clustering features standardised before K-Means (raw likes/shares dominate distance otherwise)
- LSTM module exposes a `trend_direction` helper for the dashboard (real LSTM inference needs trained weights)
- Scraper ships as a mock — real Instagram/TikTok/X scraping requires authenticated APIs and is platform-ToS gated
