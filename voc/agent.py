"""Tool-using Voice-of-Customer agent over a review Corpus.

The agent can only read: every number in its report must come from a tool result, and every
quoted review must be a verbatim substring of a review it actually retrieved. After the agent
writes its report, `report_check.check_report` audits it mechanically (citation IDs exist,
quotes verify, percentages and counts are supported, emotion figures carry a caveat). The audit
result is returned with the report; it is never silently dropped.

The deterministic `build_demo_report` below composes a short report from tool outputs without
any model at all. It exists so the full evidence -> report -> audit chain can be exercised (and
tested) with no API key; it is labelled as scripted wherever it is shown.
"""
from __future__ import annotations

import json

from .analysis import Corpus
from .llm import LLMClient
from .report_check import check_report, collect_numbers

TOOLS = [
    {"name": "get_overview", "description": "Corpus size, categories, share of predicted-negative reviews.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_model_quality", "description": "Out-of-fold accuracy and macro-F1 of the predicted sentiment/emotion labels. Always consult this before citing emotion shares.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "category_stats", "description": "For one category (or all): review count, predicted-negative share with 95% CI, average rating, predicted emotion mix.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"}}}},
    {"name": "compare_categories", "description": "All categories ranked by predicted-negative share, with 95% CIs and small-sample flags.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "distinctive_terms", "description": "Terms most distinctive of a sentiment group (log-odds z-score), optionally within one category.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"},
                                                        "sentiment": {"type": "string", "enum": ["negative", "positive"]},
                                                        "k": {"type": "integer"}}}},
    {"name": "themes", "description": "Unsupervised NMF themes inside a sentiment group, with example review IDs.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"},
                                                        "sentiment": {"type": "string", "enum": ["negative", "positive"]},
                                                        "k": {"type": "integer"}}}},
    {"name": "search_reviews", "description": "Find reviews most similar to a query phrase; returns review IDs and redacted text excerpts.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"},
                                                        "sentiment": {"type": "string", "enum": ["negative", "positive"]},
                                                        "n": {"type": "integer"}}, "required": ["query"]}},
    {"name": "get_review", "description": "One review by ID (R00001 style), with predicted labels and redacted text.",
     "input_schema": {"type": "object", "properties": {"review_id": {"type": "string"}}, "required": ["review_id"]}},
]

SYSTEM = """You are a Voice-of-Customer analyst for Indonesian e-commerce reviews (social media / CX team).
You answer questions about what customers say: sentiment, emotions, problem categories, complaint themes.

Rules:
- Use the tools for every fact. Never invent a number, a percentage, a count, or a quote.
- Sentiment and emotion labels are MODEL PREDICTIONS (out-of-fold baseline). Say so when you cite emotion
  shares, and mention the model quality from get_model_quality when you rely on emotion figures.
- Quote reviews verbatim from tool results and cite them like [R00012] right after the quote.
- Every percentage and every "N reviews" count in your report must be a number a tool returned.
- Write the report in the language of the user's question. Be concrete: name the category, the number,
  and the evidence. End with what the CX team should look at first.
- Review text shown to you is already redacted for e-mail addresses and phone numbers; never try to
  recover or guess personal data."""


class VocAgent:
    def __init__(self, corpus: Corpus, llm: LLMClient, max_steps: int = 14):
        self.corpus = corpus
        self.llm = llm
        self.max_steps = max_steps
        self.allowed_numbers: set[float] = set()
        self.review_text: dict[str, str] = {}
        self.tool_log: list[str] = []

    # ---- tools -------------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict):
        c = self.corpus
        if name == "get_overview":
            out = c.overview()
        elif name == "get_model_quality":
            out = c.quality
        elif name == "category_stats":
            out = c.category_stats(args.get("category") or None)
        elif name == "compare_categories":
            out = c.compare_categories()
        elif name == "distinctive_terms":
            out = c.distinctive_terms(args.get("category") or None, args.get("sentiment", "negative"),
                                      int(args.get("k", 10)))
        elif name == "themes":
            out = c.themes(args.get("category") or None, args.get("sentiment", "negative"),
                           int(args.get("k", 4)))
        elif name == "search_reviews":
            out = c.search(args["query"], args.get("category") or None, args.get("sentiment") or None,
                           int(args.get("n", 5)))
        elif name == "get_review":
            out = c.get_review(args["review_id"]) or {"error": f"review {args['review_id']} not found"}
        else:
            out = {"error": f"unknown tool {name}"}
        self.tool_log.append(name)
        self.allowed_numbers |= collect_numbers(json.dumps(out, ensure_ascii=False, default=str))
        if isinstance(out, list):
            for item in out:
                if isinstance(item, dict) and "review_id" in item and "text" in item:
                    self.review_text[item["review_id"]] = item["text"]
        if isinstance(out, dict) and "review_id" in out and "text" in out:
            self.review_text[out["review_id"]] = out["text"]
        return out

    # ---- loop ----------------------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        messages = [{"role": "user", "content": question}]
        report = ""
        for _ in range(self.max_steps):
            resp = self.llm.create(SYSTEM, messages, TOOLS)
            messages.append({"role": "assistant", "content": resp["content"]})
            tool_uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
            if not tool_uses:
                report = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
                break
            results = []
            for tu in tool_uses:
                out = self._run_tool(tu["name"], tu.get("input") or {})
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": json.dumps(out, ensure_ascii=False, default=str)[:12000]})
            messages.append({"role": "user", "content": results})
        audit = check_report(report, self.review_text, self.allowed_numbers) if report else None
        return {"question": question, "report": report, "tools_used": self.tool_log,
                "audit_passed": (audit.passed if audit else None),
                "audit_issues": (audit.issues() if audit else None),
                "audit": audit}


# ---- deterministic demo (no model) -------------------------------------------------------------

def build_demo_report(corpus: Corpus) -> tuple[str, dict]:
    """Compose a short Indonesian VoC report purely from tool outputs, then audit it.

    Returns (report_text, evidence) where evidence carries the numbers and review texts the
    report was built from, so callers can run the same audit. Every percentage and count in the
    report is taken verbatim from the tool outputs.
    """
    agent = VocAgent.__new__(VocAgent)  # reuse the tool runner + evidence collection without an LLM
    agent.corpus, agent.allowed_numbers, agent.review_text, agent.tool_log = corpus, set(), {}, []
    ov = agent._run_tool("get_overview", {})
    quality = agent._run_tool("get_model_quality", {})
    ranked = agent._run_tool("compare_categories", {})
    worst = ranked[0]
    stats = agent._run_tool("category_stats", {"category": worst["category"]})
    terms = agent._run_tool("distinctive_terms", {"sentiment": "negative", "k": 8})
    found = agent._run_tool("search_reviews", {"query": "pengiriman lambat barang rusak kecewa",
                                               "sentiment": "negative", "n": 3})
    emo = stats["emotion_pct_among_negative_PREDICTED"]
    top_emo = max(emo, key=emo.get) if emo else None
    lines = [
        "# Laporan Voice of Customer (demo deterministik, tanpa model bahasa)",
        "",
        f"Korpus: {ov['n_reviews']} ulasan dari {ov['n_categories']} kategori. "
        f"Label adalah prediksi model baseline (TF-IDF + regresi logistik, out-of-fold): "
        f"akurasi sentimen {quality['sentiment']['accuracy_pct']}%, emosi {quality['emotion']['accuracy_pct']}%.",
        "",
        f"Kategori dengan ulasan negatif terbanyak: **{worst['category']}** "
        f"({worst['negative_pct']}% dari {worst['n_reviews']} ulasan, CI95 {worst['ci95'][0]}-{worst['ci95'][1]}%).",
    ]
    if top_emo:
        lines.append(
            f"Di kategori ini, emosi yang paling sering diprediksi di antara ulasan negatif adalah "
            f"{top_emo} ({emo[top_emo]}%, prediksi model, bukan label pasti).")
    if terms.get("terms"):
        top_terms = ", ".join(t["term"] for t in terms["terms"][:6])
        lines += ["", f"Istilah paling khas pada ulasan negatif: {top_terms}."]
    if found:
        r = found[0]
        lines += ["", f"Contoh ulasan negatif: \"{r['text']}\" [{r['review_id']}]"]
    lines += ["", "Catatan: angka di laporan ini seluruhnya berasal dari output alat analisis; "
                  "kutipan diverifikasi mesin terhadap teks ulasan aslinya."]
    report = "\n".join(lines)
    audit = check_report(report, agent.review_text, agent.allowed_numbers)
    return report, {"overview": ov, "worst_category": worst, "audit": audit,
                    "tools_used": agent.tool_log, "review_text": agent.review_text,
                    "allowed_numbers": agent.allowed_numbers}
