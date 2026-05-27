"""BERT-based sentiment + trend category tagging for captions/comments.

The HuggingFace `pipeline("sentiment-analysis")` defaults to a distilBERT model
fine-tuned on SST-2 — good enough for positive/negative classification of
short fashion captions. Trend category is rule-matched from a keyword map.
"""

from functools import lru_cache
from typing import Dict

TREND_KEYWORDS = {
    "ankara_fusion": ["ankara", "owambe", "senator", "agbada", "aso ebi"],
    "alte": ["alte", "alté", "indie", "vintage"],
    "streetwear": ["streetwear", "hoodie", "sneakers", "drip", "campus"],
    "corporate_chic": ["corporate", "internship", "office", "blazer", "suit"],
    "casual": ["casual", "minimal", "plain", "lazy", "basic"],
}


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline
    return pipeline("sentiment-analysis")


def analyse_sentiment(text: str) -> Dict:
    clf = get_classifier()
    result = clf(text)[0]
    return {"label": result["label"], "score": float(result["score"])}


def detect_trend_category(text: str) -> str:
    lowered = text.lower()
    for category, keywords in TREND_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return category
    return "uncategorised"


def analyse(text: str) -> Dict:
    sentiment = analyse_sentiment(text)
    return {
        "text": text,
        "sentiment": sentiment["label"],
        "confidence": sentiment["score"],
        "trend_category": detect_trend_category(text),
    }


if __name__ == "__main__":
    samples = [
        "This Ankara combo is trending everywhere on campus",
        "I hate how boring this outfit looks",
        "Alte vibes for the weekend",
    ]
    for s in samples:
        print(analyse(s))
