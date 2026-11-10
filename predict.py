"""Predict sentiment for custom text using the trained model.

Usage:
    python predict.py "This movie was absolutely fantastic!"
    python predict.py            # runs built-in examples
"""
import sys

import joblib

from train import OUT, clean_text

EXAMPLES = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "Terrible plot, bad acting, and a complete waste of time.",
    "The visuals were great but the story was boring and far too long.",
]


def predict(model, text: str):
    proba = model.predict_proba([clean_text(text)])[0]
    label = "Positive" if proba[1] >= 0.5 else "Negative"
    return label, max(proba)


if __name__ == "__main__":
    model = joblib.load(OUT / "sentiment_model.joblib")
    texts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLES
    for t in texts:
        label, conf = predict(model, t)
        print(f"{label} ({conf:.0%}) -> {t}")
