"""Audit a written report against the evidence it was built from.

Checks (all mechanical, none needs a model):
  invalid_ids          cited [R00012] that do not exist in the corpus
  unverified_quotes    quoted text (followed by a citation) that is not a verbatim substring of a cited review
  uncited_quotes       long quoted text with no citation
  unsupported_numbers  percentages / review counts that no tool returned
  emotion_without_caveat  emotion shares cited without any hedge that they are model predictions
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = r"\[(R\d{5}(?:\s*,\s*R\d{5})*)\]"
QUOTE_CITED = re.compile(r"[\"“«]([^\"“”«»]{6,300})[\"”»]\s*\(?\s*" + CITE)
QUOTE_ANY = re.compile(r"[\"“«]([^\"“”«»]{20,300})[\"”»]")
PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")
COUNT = re.compile(r"(\d[\d.,]*)\s+(?:ulasan|review|reviews)\b", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")
EMOTION_WORDS = re.compile(r"\b(marah|anger|takut|fear|sedih|sadness|senang|happy|cinta|love|emosi|emotion)\b", re.I)
HEDGE = re.compile(r"(prediksi|diprediksi|perkiraan|estimasi|estimate|predicted|akurasi|tidak sepenuhnya|kurang andal|keandalan)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def collect_numbers(text: str) -> set[float]:
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(float(m.replace(",", ".")))
        except ValueError:
            pass
    return out


@dataclass
class ReportCheck:
    invalid_ids: list = field(default_factory=list)
    unverified_quotes: list = field(default_factory=list)
    uncited_quotes: list = field(default_factory=list)
    unsupported_numbers: list = field(default_factory=list)
    emotion_without_caveat: bool = False
    cited_ids: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.invalid_ids or self.unverified_quotes or self.uncited_quotes or self.unsupported_numbers
                    or self.emotion_without_caveat)

    def issues(self) -> int:
        return (len(self.invalid_ids) + len(self.unverified_quotes) + len(self.uncited_quotes) + len(self.unsupported_numbers)
                + int(self.emotion_without_caveat))


def check_report(report: str, review_text: dict[str, str], allowed_numbers: set[float], tol: float = 0.051) -> ReportCheck:
    res = ReportCheck()
    ids = []
    for grp in re.findall(CITE, report):
        ids += [x.strip() for x in grp.split(",")]
    res.cited_ids = sorted(set(ids))
    res.invalid_ids = [i for i in res.cited_ids if i not in review_text]

    cited_spans = []
    for m in QUOTE_CITED.finditer(report):
        cited_spans.append(m.span(1))
        q = m.group(1)
        frags = [f for f in re.split(r"\.{3}|…", q) if len(_norm(f)) >= 6]
        pool = [_norm(review_text[i.strip()]) for i in m.group(2).split(",") if i.strip() in review_text]
        if not pool or not all(any(_norm(f) in p for p in pool) for f in frags):
            res.unverified_quotes.append(q)
    for m in QUOTE_ANY.finditer(report):
        if m.span(1) not in cited_spans:
            res.uncited_quotes.append(m.group(1))

    for m in PCT.finditer(report):
        v = float(m.group(1).replace(",", "."))
        if not any(abs(v - a) <= tol for a in allowed_numbers):
            res.unsupported_numbers.append(m.group(0))
    for m in COUNT.finditer(report):
        raw = m.group(1).replace(".", "").replace(",", "") if re.fullmatch(r"\d{1,3}([.,]\d{3})+", m.group(1)) else m.group(1).replace(",", ".")
        if float(raw) not in allowed_numbers:
            res.unsupported_numbers.append(m.group(0))

    emo_pct = any(EMOTION_WORDS.search(line) and PCT.search(line) for line in report.splitlines())
    res.emotion_without_caveat = bool(emo_pct and not HEDGE.search(report))
    return res
