"""Command line: deterministic analysis, a scripted demo, or the live agent.

  python -m voc.cli analyze "data/PRDECT-ID Dataset.csv"
  python -m voc.cli analyze "data/PRDECT-ID Dataset.csv" --agent --question "Kategori apa yang paling bermasalah?"
  python -m voc.cli demo            # scripted questions + an audited report, no API key needed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import Corpus
from .data import load_reviews

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "PRDECT-ID Dataset.csv"


def _load(path: str) -> Corpus:
    df, info = load_reviews(path)
    print(f"Loaded {info['rows_kept']} reviews (separator {info['separator']!r}, "
          f"columns: {', '.join(info['columns_found'])})")
    print("Building predicted labels (out-of-fold TF-IDF + logistic regression) ...")
    return Corpus(df)


def cmd_analyze(args) -> int:
    corpus = _load(args.csv)
    ov = corpus.overview()
    print(f"\n{ov['n_reviews']} reviews, {ov['n_categories']} categories, "
          f"predicted negative: {ov['predicted_negative_pct']}%")
    for task, q in corpus.quality.items():
        print(f"Model quality [{task}]: accuracy {q['accuracy_pct']}%, macro-F1 {q['macro_f1_pct']}% "
              f"(out-of-fold, n={q['n_labelled']})")
    print("\nCategories with the most predicted-negative reviews:")
    for row in corpus.compare_categories()[:5]:
        flag = " (small sample)" if row["small_sample"] else ""
        print(f"  {row['category']:<28} {row['negative_pct']:>5}% negative of {row['n_reviews']:>5} "
              f"CI95 {row['ci95'][0]}-{row['ci95'][1]}{flag}")
    terms = corpus.distinctive_terms(sentiment="negative", k=10)
    if terms.get("terms"):
        print("\nMost distinctive terms in negative reviews:")
        print("  " + ", ".join(t["term"] for t in terms["terms"]))
    if args.agent:
        from .agent import VocAgent
        from .llm import AnthropicClient
        agent = VocAgent(corpus, AnthropicClient())
        res = agent.ask(args.question)
        print("\n" + res["report"])
        print(f"\n[audit] passed={res['audit_passed']} issues={res['audit_issues']} "
              f"tools={', '.join(res['tools_used'])}")
    return 0


def cmd_demo(args) -> int:
    from .agent import build_demo_report
    corpus = _load(args.csv)
    print("\n=== Scripted demo on the real PRDECT-ID data (no API key) ===")
    print("\nQ1: Kategori apa yang paling banyak ulasan negatifnya?")
    for row in corpus.compare_categories()[:3]:
        print(f"  {row['category']:<28} {row['negative_pct']}% negative ({row['n_reviews']} reviews)")
    print("\nQ2: Apa istilah yang paling khas dari ulasan negatif?")
    terms = corpus.distinctive_terms(sentiment="negative", k=8)
    print("  " + ", ".join(t["term"] for t in terms["terms"]))
    print("\nQ3: Tunjukkan contoh ulasan negatif tentang pengiriman.")
    for r in corpus.search("pengiriman lambat paket belum sampai", sentiment="negative", n=3):
        print(f"  [{r['review_id']}] ({r['category']}) {r['text'][:110]}")
    report, ev = build_demo_report(corpus)
    print("\n=== Composed report ===\n")
    print(report)
    audit = ev["audit"]
    print(f"\n[audit] passed={audit.passed} issues={audit.issues()} "
          f"(invalid_ids={len(audit.invalid_ids)}, unverified_quotes={len(audit.unverified_quotes)}, "
          f"unsupported_numbers={len(audit.unsupported_numbers)}, "
          f"emotion_without_caveat={audit.emotion_without_caveat})")
    return 0 if audit.passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="voc.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("csv")
    pa.add_argument("--agent", action="store_true")
    pa.add_argument("--question", default="Kategori apa yang paling bermasalah dan apa buktinya?")
    pa.set_defaults(fn=cmd_analyze)
    pd_ = sub.add_parser("demo")
    pd_.add_argument("--csv", default=str(DEFAULT_CSV))
    pd_.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
