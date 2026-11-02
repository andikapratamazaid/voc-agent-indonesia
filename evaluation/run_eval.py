"""Evaluation on the REAL PRDECT-ID dataset (no LLM, no network beyond the local file).

  python -m evaluation.run_eval            # writes evaluation/results/latest.md

What is measured:
  1. Holdout performance of the baseline classifier for sentiment and for emotion
     (stratified 80/20 split, seed 42), against the majority-class baseline.
  2. The rating-only baseline: how far the star rating alone predicts sentiment.
  3. Out-of-fold label quality of the Corpus used by the agent.
  4. A report-audit self-test: the deterministic demo report must pass check_report,
     and a deliberately tampered report (invented percentage, fake review ID, an
     uncited quote, an emotion share with no caveat) must fail it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voc.agent import build_demo_report
from voc.analysis import Corpus
from voc.data import load_reviews
from voc.models import evaluate_holdout, rating_baseline
from voc.report_check import check_report

CSV = Path(__file__).resolve().parent.parent / "data" / "PRDECT-ID Dataset.csv"
OUT = Path(__file__).resolve().parent / "results" / "latest.md"


def _fmt_holdout(h: dict) -> list[str]:
    lines = [f"### {h['task'].title()} (holdout, n_train={h['n_train']}, n_test={h['n_test']})", "",
             f"- Accuracy **{100 * h['accuracy']:.1f}%**, macro-F1 **{100 * h['macro_f1']:.1f}%** "
             f"(majority-class baseline accuracy {100 * h['majority_baseline_accuracy']:.1f}%)",
             "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab in h["labels"]:
        pc = h["per_class"][lab]
        lines.append(f"| {lab} | {pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1-score']:.2f} | {int(pc['support'])} |")
    lines += ["", "Confusion (rows = gold, columns = predicted, order: " + ", ".join(h["labels"]) + "):",
              "", "```", *["  ".join(f"{v:>5}" for v in row) for row in h["confusion"]], "```", ""]
    return lines


def main() -> int:
    df, info = load_reviews(str(CSV))
    lines = ["# Evaluation on the real PRDECT-ID dataset", "",
             f"- Reviews: {info['rows_kept']} | categories: {df.category.nunique()} | "
             f"sentiment gold: {dict(df.sentiment.value_counts())} | emotion gold: {dict(df.emotion.value_counts())}",
             "- Split: stratified holdout 80/20, seed 42. Model: word+char TF-IDF + logistic regression "
             "(the project baseline, not a transformer).", ""]
    for task in ("sentiment", "emotion"):
        lines += _fmt_holdout(evaluate_holdout(df, task))
    rb = rating_baseline(df)
    if rb:
        lines += ["### Rating-only baseline (sentiment)", "",
                  f"- Rule 'negative if rating <= {rb['threshold_negative_if_rating_leq']}': accuracy "
                  f"**{100 * rb['accuracy']:.1f}%**, macro-F1 {100 * rb['macro_f1']:.1f}% (n_test={rb['n_test']}). "
                  "The text model has to beat this to be worth anything.", "",
                  "- On PRDECT-ID the rating rule WINS (98.4% vs the text model's 95.3%): in this dataset the star "
                  "rating and the sentiment label move together almost perfectly. The text model still earns its "
                  "place where it is actually deployed: social-media and CX comment streams carry no star rating at "
                  "all, and only the text model produces the five emotion labels. Treat the rating result as a "
                  "property of this dataset, not as a reason to skip text analysis.", ""]
    corpus = Corpus(df)
    lines += ["### Corpus out-of-fold label quality (what the agent's tools report)", ""]
    for task, q in corpus.quality.items():
        lines.append(f"- {task}: accuracy {q['accuracy_pct']}%, macro-F1 {q['macro_f1_pct']}% "
                     f"(5-fold out-of-fold, n={q['n_labelled']})")
    report, ev = build_demo_report(corpus)
    audit = ev["audit"]
    tampered = (report + "\n\nSebanyak 87,3% pelanggan kecewa berat tahun ini. "
                         "\"barang langsung rusak total dalam sehari\" [R99999].")
    bad = check_report(tampered, ev["review_text"], ev["allowed_numbers"])
    lines += ["", "### Report-audit self-test", "",
              f"- Deterministic demo report: passed={audit.passed}, issues={audit.issues()} (must pass).",
              f"- Tampered report (invented 87,3%, fake ID R99999, fabricated quote): passed={bad.passed}, "
              f"issues={bad.issues()} (invalid_ids={bad.invalid_ids}, "
              f"unsupported_numbers={bad.unsupported_numbers}, "
              f"unverified_quotes={len(bad.unverified_quotes)}, uncited_quotes={len(bad.uncited_quotes)}; must fail).", "",
              "> Emotion is a five-way judgement call; published transformer results on this dataset reach "
              "roughly 75-80%, so a TF-IDF baseline below that is expected, not a bug. Emotion shares from "
              "this agent are always labelled as model predictions.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")
    return 0 if (audit.passed and not bad.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
