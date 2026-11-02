"""Baseline classifiers (TF-IDF + logistic regression) for sentiment and emotion, with honest evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .textprep import normalize

SEED = 42


def make_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0))])


def _labelled(df: pd.DataFrame, task: str) -> pd.DataFrame:
    d = df[df[task].notna()]
    if d.empty:
        raise ValueError(f"No gold labels for '{task}' in this data.")
    return d


def evaluate_holdout(df: pd.DataFrame, task: str, test_size: float = 0.2) -> dict:
    d = _labelled(df, task)
    tr, te = train_test_split(d, test_size=test_size, stratify=d[task], random_state=SEED)
    model = make_pipeline().fit(tr.text, tr[task])
    pred = model.predict(te.text)
    labels = sorted(d[task].unique())
    return {"task": task, "n_train": len(tr), "n_test": len(te), "accuracy": accuracy_score(te[task], pred),
            "macro_f1": f1_score(te[task], pred, average="macro"),
            "majority_baseline_accuracy": float(te[task].value_counts(normalize=True).max()),
            "per_class": classification_report(te[task], pred, output_dict=True, zero_division=0),
            "labels": labels, "confusion": confusion_matrix(te[task], pred, labels=labels).tolist()}


def out_of_fold(df: pd.DataFrame, task: str, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray, list]:
    """Predictions for every review from models that never saw that review (no leakage when the same corpus is analysed)."""
    d = _labelled(df, task)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_pipeline(), d.text, d[task], cv=skf, method="predict_proba")
    classes = sorted(d[task].unique())
    return np.array(classes)[proba.argmax(1)], proba, classes


def rating_baseline(df: pd.DataFrame) -> dict | None:
    """How well can the star rating alone predict sentiment? The text model must beat this to be worth anything."""
    d = df[df.sentiment.notna() & df.rating.notna()]
    if len(d) < 50:
        return None
    tr, te = train_test_split(d, test_size=0.2, stratify=d.sentiment, random_state=SEED)
    best = max(range(1, 5), key=lambda t: accuracy_score(tr.sentiment, np.where(tr.rating <= t, "negative", "positive")))
    pred = np.where(te.rating <= best, "negative", "positive")
    return {"threshold_negative_if_rating_leq": best, "accuracy": accuracy_score(te.sentiment, pred),
            "macro_f1": f1_score(te.sentiment, pred, average="macro"), "n_test": len(te)}
