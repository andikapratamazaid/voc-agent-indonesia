"""Deterministic analysis over a review corpus: predicted labels, category statistics, distinctive terms, themes, search."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel

from .data import EMOTIONS, NO_CATEGORY
from .models import make_pipeline, out_of_fold
from .textprep import redact, tokens_for_display


def analyzer(text) -> list[str]:
    toks = tokens_for_display(text)
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


class Corpus:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.df["text_view"] = self.df.text.map(redact)
        self.quality: dict = {}
        for task, out in (("sentiment", "pred_sentiment"), ("emotion", "pred_emotion")):
            self._predict(task, out)
        self._tfidf = TfidfVectorizer(analyzer=analyzer, min_df=1, sublinear_tf=True)
        self._mat = self._tfidf.fit_transform(self.df.text)
        self.by_id = {r.review_id: i for i, r in enumerate(self.df.itertuples())}

    # ---- labels -----------------------------------------------------------------------------------
    def _predict(self, task: str, out: str) -> None:
        d = self.df
        lab = d[task].notna() if task in d else pd.Series(False, index=d.index)
        if lab.sum() < 50 or d.loc[lab, task].nunique() < 2:
            raise ValueError(f"Need at least 50 labelled reviews with 2+ classes for '{task}' to build predicted labels.")
        pred, _, _ = out_of_fold(d, task)                                  # out-of-fold: no review is scored by a model that saw it
        d[out] = None
        d.loc[lab, out] = pred
        if (~lab).any():
            d.loc[~lab, out] = make_pipeline().fit(d.loc[lab, "text"], d.loc[lab, task]).predict(d.loc[~lab, "text"])
        gold = d.loc[lab, task]
        self.quality[task] = {
            "accuracy_pct": round(100 * accuracy_score(gold, pred), 1),
            "macro_f1_pct": round(100 * f1_score(gold, pred, average="macro"), 1),
            "per_class_f1_pct": {c: round(100 * f1_score(gold == c, pred == c), 1) for c in sorted(gold.unique())},
            "n_labelled": int(lab.sum()), "method": "TF-IDF + logistic regression, 5-fold out-of-fold",
        }

    # ---- selection --------------------------------------------------------------------------------
    def _sel(self, category: str | None = None, sentiment: str | None = None) -> pd.DataFrame:
        d = self.df
        if category and category != NO_CATEGORY:
            d = d[d.category.str.lower() == category.lower()]
        if sentiment:
            d = d[d.pred_sentiment == sentiment]
        return d

    def categories(self) -> list[str]:
        return sorted(self.df.category.unique())

    # ---- statistics -------------------------------------------------------------------------------
    def overview(self) -> dict:
        d = self.df
        return {"n_reviews": int(len(d)), "n_categories": int(d.category.nunique()), "categories": self.categories()[:40],
                "has_rating": bool(d.rating.notna().any()),
                "predicted_negative_pct": round(100 * (d.pred_sentiment == "negative").mean(), 1),
                "label_source": "predicted (out-of-fold) by a baseline model; see get_model_quality"}

    def category_stats(self, category: str | None = None) -> dict:
        d = self._sel(category)
        if d.empty:
            return {"error": f"no reviews for category '{category}'", "categories": self.categories()[:40]}
        n, neg = len(d), int((d.pred_sentiment == "negative").sum())
        lo, hi = wilson(neg, n)
        emo_all = {e: round(100 * (d.pred_emotion == e).mean(), 1) for e in EMOTIONS}
        negd = d[d.pred_sentiment == "negative"]
        emo_neg = {e: round(100 * (negd.pred_emotion == e).mean(), 1) for e in EMOTIONS} if len(negd) else {}
        return {"category": category or NO_CATEGORY, "n_reviews": int(n), "n_negative": neg,
                "negative_pct": round(100 * neg / n, 1), "negative_pct_ci95": [round(lo, 1), round(hi, 1)],
                "avg_rating": None if d.rating.isna().all() else round(float(d.rating.mean()), 2),
                "emotion_pct_all_reviews_PREDICTED": emo_all, "emotion_pct_among_negative_PREDICTED": emo_neg,
                "small_sample": bool(n < 30)}

    def compare_categories(self, min_n: int = 20) -> list[dict]:
        rows = []
        for c, g in self.df.groupby("category"):
            n, neg = len(g), int((g.pred_sentiment == "negative").sum())
            lo, hi = wilson(neg, n)
            rows.append({"category": c, "n_reviews": n, "negative_pct": round(100 * neg / n, 1),
                         "ci95": [round(lo, 1), round(hi, 1)], "small_sample": n < min_n})
        return sorted(rows, key=lambda r: -r["negative_pct"])

    # ---- terms and themes ----------------------------------------------------------------------------
    def distinctive_terms(self, category: str | None = None, sentiment: str = "negative", k: int = 10) -> dict:
        fg = self._sel(category, sentiment)
        if category and category != NO_CATEGORY:
            bg = self.df[(self.df.pred_sentiment == sentiment) & (self.df.category.str.lower() != category.lower())]
            against = f"{sentiment} reviews of other categories"
        else:
            bg = self.df[self.df.pred_sentiment != sentiment]
            against = f"reviews with the opposite sentiment"
        if len(fg) < 10 or len(bg) < 10:
            return {"error": "too few reviews for a reliable comparison", "n_selected": int(len(fg))}
        cv = CountVectorizer(analyzer=analyzer, binary=True, min_df=3)
        X = cv.fit_transform(pd.concat([fg.text, bg.text]))
        a = np.asarray(X[: len(fg)].sum(0)).ravel(); b = np.asarray(X[len(fg):].sum(0)).ravel()
        nf, nb = len(fg), len(bg)
        lo = np.log((a + .5) / (nf - a + .5)) - np.log((b + .5) / (nb - b + .5))
        z = lo / np.sqrt(1 / (a + .5) + 1 / (nf - a + .5) + 1 / (b + .5) + 1 / (nb - b + .5))
        terms = np.array(cv.get_feature_names_out())
        order = [i for i in np.argsort(-z) if a[i] >= 3][:k]
        return {"category": category or NO_CATEGORY, "sentiment": sentiment, "n_reviews": int(nf), "compared_against": against,
                "terms": [{"term": terms[i], "n_reviews": int(a[i]), "share_pct": round(100 * a[i] / nf, 1), "z": round(float(z[i]), 1)} for i in order]}

    def themes(self, category: str | None = None, sentiment: str = "negative", k: int = 4) -> dict:
        d = self._sel(category, sentiment)
        k = min(k, len(d) // 8)
        if k < 2:
            return {"error": "too few reviews to find themes", "n_selected": int(len(d))}
        tv = TfidfVectorizer(analyzer=analyzer, min_df=2, sublinear_tf=True)
        X = tv.fit_transform(d.text)
        if X.shape[1] < k:
            return {"error": "vocabulary too small for themes", "n_selected": int(len(d))}
        nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
        W = nmf.fit_transform(X)
        terms = np.array(tv.get_feature_names_out())
        assign = W.argmax(1)
        out = []
        for t in range(k):
            members = np.where((assign == t) & (W.max(1) > 0))[0]
            top = [str(x) for x in terms[np.argsort(-nmf.components_[t])[:6]]]
            ex = members[np.argsort(-W[members, t])][:3]
            out.append({"theme": t + 1, "top_terms": top, "n_reviews": int(len(members)),
                        "share_pct": round(100 * len(members) / len(d), 1), "example_review_ids": [d.iloc[i].review_id for i in ex]})
        return {"category": category or NO_CATEGORY, "sentiment": sentiment, "n_reviews": int(len(d)),
                "themes": sorted(out, key=lambda x: -x["n_reviews"]),
                "note": "Themes are unsupervised word groups; a human or the report must interpret them."}

    # ---- retrieval -------------------------------------------------------------------------------------
    def search(self, query: str, category: str | None = None, sentiment: str | None = None, n: int = 5) -> list[dict]:
        q = self._tfidf.transform([query])
        sims = linear_kernel(q, self._mat).ravel()
        mask = np.zeros(len(self.df), bool)
        mask[self._sel(category, sentiment).index.to_numpy()] = True
        sims = np.where(mask, sims, -1)
        res = []
        for i in np.argsort(-sims)[:max(1, min(n, 10))]:
            if sims[i] <= 0:
                break
            r = self.df.iloc[i]
            res.append({"review_id": r.review_id, "score": round(float(sims[i]), 3), "category": r.category,
                        "sentiment_pred": r.pred_sentiment, "emotion_pred": r.pred_emotion, "text": r.text_view[:240]})
        return res

    def get_review(self, review_id: str) -> dict | None:
        i = self.by_id.get(review_id)
        if i is None:
            return None
        r = self.df.iloc[i]
        return {"review_id": r.review_id, "category": r.category, "rating": None if pd.isna(r.rating) else float(r.rating),
                "sentiment_pred": r.pred_sentiment, "emotion_pred": r.pred_emotion, "text": r.text_view}
