"""Support-ticket route classifier, baseline.

Handed over from a previous engineer along with eval_report.md.
Trains on data/train.csv and reports how well it does.

    python3 baseline/baseline_classifier.py
"""
import csv
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
ROUTES = ["account-access", "transaction-dispute", "fraud-report", "general"]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def main():
    texts, labels = load(DATA)
    texts = np.array(texts)
    labels = np.array(labels)
    print(f"loaded {len(texts)} rows")

    # Stratified 5-fold CV: every fold keeps the class ratios, and the
    # vectorizer is fit on each fold's train rows only (no leakage from
    # fitting TF-IDF on the held-out rows, unlike the original single split).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_true, y_pred = [], []

    for train_idx, test_idx in skf.split(texts, labels):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X_train = vectorizer.fit_transform(texts[train_idx])
        X_test = vectorizer.transform(texts[test_idx])

        clf = LogisticRegression(max_iter=2000, C=10.0)
        clf.fit(X_train, labels[train_idx])

        y_true.extend(labels[test_idx])
        y_pred.extend(clf.predict(X_test))

    acc = accuracy_score(y_true, y_pred)
    print(f"cross-validated accuracy (5-fold): {acc:.4f}")
    print()
    print("per-class precision/recall/F1 (this is the number that matters for")
    print("fraud-report, since accuracy alone hides minority-class misses):")
    print(classification_report(y_true, y_pred, labels=ROUTES))
    print("confusion matrix (rows=true, cols=predicted), order:", ROUTES)
    print(confusion_matrix(y_true, y_pred, labels=ROUTES))
    return acc


_fitted = None


def _get_fitted():
    global _fitted
    if _fitted is None:
        texts, labels = load(DATA)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = vectorizer.fit_transform(texts)
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(X, labels)
        _fitted = (vectorizer, clf)
    return _fitted


def predict(text):
    """predict(text) -> route label."""
    vectorizer, clf = _get_fitted()
    return clf.predict(vectorizer.transform([text]))[0]


if __name__ == "__main__":
    main()
