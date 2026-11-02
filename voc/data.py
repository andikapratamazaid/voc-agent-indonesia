"""Load a PRDECT-ID style review file (semicolon- or comma-delimited CSV) into a clean DataFrame.

PRDECT-ID is documented as a semicolon-delimited UTF-8 CSV whose text column is 'Customer Review', with sentiment
(Positive/Negative) and emotion (Happy, Love, Sad, Fear, Anger) labels. Other columns (category, rating, price ...)
are used when present. Column names are matched by alias because I could not verify the exact header of every
published copy; the loader reports what it found.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

ALIASES = {
    "text": ["customer review", "review", "ulasan", "text", "teks"],
    "sentiment": ["sentiment", "sentimen"],
    "emotion": ["emotion", "emosi"],
    "rating": ["customer rating", "rating"],
    "category": ["category", "kategori", "product category", "kategori produk"],
    "product": ["product name", "nama produk", "product"],
    "price": ["price", "harga"],
}
SENTIMENT_MAP = {"positive": "positive", "pos": "positive", "positif": "positive", "negative": "negative",
                 "neg": "negative", "negatif": "negative"}
EMOTION_MAP = {"happy": "happy", "happiness": "happy", "senang": "happy", "bahagia": "happy", "love": "love", "cinta": "love",
               "sad": "sadness", "sadness": "sadness", "sedih": "sadness", "fear": "fear", "takut": "fear",
               "anger": "anger", "angry": "anger", "marah": "anger"}
EMOTIONS = ["happy", "love", "sadness", "fear", "anger"]
SENTIMENTS = ["negative", "positive"]
NO_CATEGORY = "(semua)"


def _detect_sep(path: Path) -> str:
    head = path.read_text(encoding="utf-8-sig", errors="replace")[:5000]
    try:
        return csv.Sniffer().sniff(head.split("\n")[0] + "\n", delimiters=";,\t").delimiter
    except csv.Error:
        return ";" if head.count(";") > head.count(",") else ","


def load_reviews(path: str, sep: str | None = None) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to get PRDECT-ID.")
    sep = sep or _detect_sep(p)
    raw = pd.read_csv(p, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in raw.columns}
    found = {k: next((cols[a] for a in names if a in cols), None) for k, names in ALIASES.items()}
    if not found["text"]:
        raise ValueError(f"No review-text column found. Columns present: {list(raw.columns)}")
    df = pd.DataFrame({"text": raw[found["text"]].astype(str).str.strip()})
    df["sentiment"] = (raw[found["sentiment"]].str.strip().str.lower().map(SENTIMENT_MAP) if found["sentiment"] else None)
    df["emotion"] = (raw[found["emotion"]].str.strip().str.lower().map(EMOTION_MAP) if found["emotion"] else None)
    df["rating"] = pd.to_numeric(raw[found["rating"]], errors="coerce") if found["rating"] else float("nan")
    df["category"] = raw[found["category"]].str.strip().replace("", NO_CATEGORY) if found["category"] else NO_CATEGORY
    df["product"] = raw[found["product"]].str.strip() if found["product"] else ""
    n_before = len(df)
    df = df[df.text.str.len() > 0].reset_index(drop=True)
    df.insert(0, "review_id", [f"R{i + 1:05d}" for i in range(len(df))])
    info = {"path": str(p), "separator": sep, "rows_read": n_before, "rows_kept": len(df), "columns_found": {k: v for k, v in found.items() if v},
            "columns_missing": [k for k, v in found.items() if not v],
            "unmapped_sentiment": int(df.sentiment.isna().sum()) if found["sentiment"] else None,
            "unmapped_emotion": int(df.emotion.isna().sum()) if found["emotion"] else None}
    return df, info
