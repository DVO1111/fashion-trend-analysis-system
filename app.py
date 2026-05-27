"""Flask dashboard that ties the multimodal pipeline together."""

import os
import re
import tempfile
import pandas as pd
from flask import Flask, render_template, request, jsonify

import text_model
import clustering
import forecast

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ENGAGEMENT_CSV = "dataset/engagement.csv"
CNN_REPORT_PATH = "models/cnn_report.txt"
CNN_LABELS_PATH = "models/cnn_labels.json"
CNN_MODEL_PATH = "models/cnn_model.h5"


def _read_cnn_accuracy():
    """Parse the overall validation accuracy from the saved CNN report."""
    if not os.path.exists(CNN_REPORT_PATH):
        return None
    with open(CNN_REPORT_PATH) as f:
        text = f.read()
    match = re.search(r"Overall validation accuracy:\s*([\d.]+)%", text)
    return float(match.group(1)) if match else None


def _stats():
    """All numbers shown on the dashboard. Every value is computed from
    actual project files, not invented for display."""
    out = {
        "engagement_records": 0,
        "unique_trends": 0,
        "total_likes": 0,
        "total_comments": 0,
        "total_shares": 0,
        "cnn_accuracy_pct": _read_cnn_accuracy(),
        "cnn_trained": os.path.exists(CNN_MODEL_PATH),
        "cnn_classes": [],
        "hashtags_tracked": 0,
        "captions_tracked": 0,
    }
    if os.path.exists(ENGAGEMENT_CSV):
        df = pd.read_csv(ENGAGEMENT_CSV)
        out["engagement_records"] = int(len(df))
        out["unique_trends"] = int(df["trend"].nunique())
        out["total_likes"] = int(df["likes"].sum())
        out["total_comments"] = int(df["comments"].sum())
        out["total_shares"] = int(df["shares"].sum())
    if os.path.exists(CNN_LABELS_PATH):
        import json
        with open(CNN_LABELS_PATH) as f:
            out["cnn_classes"] = json.load(f)
    if os.path.exists("dataset/hashtags.csv"):
        out["hashtags_tracked"] = int(len(pd.read_csv("dataset/hashtags.csv")))
    if os.path.exists("dataset/captions.csv"):
        out["captions_tracked"] = int(len(pd.read_csv("dataset/captions.csv")))
    return out

# Warm the BERT pipeline at import time. With gunicorn --preload this happens
# once in the master process before workers fork, so each worker inherits the
# loaded model via copy-on-write and the first user request is instant.
if os.environ.get("WARMUP_BERT", "1") == "1":
    text_model.get_classifier()


def _engagement_summary():
    if not os.path.exists(ENGAGEMENT_CSV):
        return []
    df = pd.read_csv(ENGAGEMENT_CSV)
    agg = (
        df.groupby("trend")[["likes", "comments", "shares"]]
        .sum()
        .sort_values("likes", ascending=False)
        .reset_index()
    )
    return agg.to_dict(orient="records")


@app.route("/")
def home():
    return render_template(
        "index.html",
        trends=_engagement_summary(),
        stats=_stats(),
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(_stats())


@app.route("/api/analyse-text", methods=["POST"])
def api_analyse_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    return jsonify(text_model.analyse(text))


@app.route("/api/analyse-image", methods=["POST"])
def api_analyse_image():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    from image_model import predict_style, load, build_model, load_labels, MODEL_PATH

    file = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        labels = load_labels()
        if os.path.exists(MODEL_PATH):
            model = load(MODEL_PATH)
            trained = True
        else:
            model = build_model(len(labels))
            trained = False
        result = predict_style(model, tmp_path, labels)
        result["trained"] = trained
        return jsonify(result)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/cluster", methods=["GET"])
def api_cluster():
    if not os.path.exists(ENGAGEMENT_CSV):
        return jsonify({"error": "engagement.csv not found"}), 404
    df = pd.read_csv(ENGAGEMENT_CSV)
    bundle = clustering.fit(df, n_clusters=min(4, len(df)))
    labels = clustering.predict(bundle, df).tolist()
    df_out = df.copy()
    df_out["cluster"] = labels
    return jsonify(df_out.to_dict(orient="records"))


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    payload = request.get_json(silent=True) or {}
    series = payload.get("series") or []
    if len(series) < 2:
        return jsonify({"error": "series must contain at least 2 points"}), 400
    return jsonify({
        "trend_direction": forecast.trend_direction(series),
        "points": len(series),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
