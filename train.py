"""Sentiment Analysis on IMDB movie reviews: TF-IDF + Logistic Regression."""
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Remove HTML tags and non-letters, lowercase, collapse spaces."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def main():
    print("Loading IMDB dataset...")
    data = load_dataset("stanfordnlp/imdb")
    X_train = [clean_text(t) for t in data["train"]["text"]]
    y_train = data["train"]["label"]
    X_test = [clean_text(t) for t in data["test"]["text"]]
    y_test = data["test"]["label"]
    print(f"Train: {len(X_train)} reviews | Test: {len(X_test)} reviews")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
                ),
            ),
            ("clf", LogisticRegression(C=4, max_iter=1000)),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.4f}")
    print("\n", classification_report(y_test, preds, target_names=["negative", "positive"]))

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["negative", "positive"], cmap="Blues"
    )
    plt.title("Confusion Matrix - IMDB Test Set")
    plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Most influential words
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    words = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    print("Top negative words:", list(words[order[:10]]))
    print("Top positive words:", list(words[order[-10:]]))

    joblib.dump(model, OUT / "sentiment_model.joblib")
    print("Saved model and results to outputs/")


if __name__ == "__main__":
    main()
