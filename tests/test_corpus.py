import unittest

from tests.helpers import fixture_df
from voc.analysis import Corpus


class TestCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus(fixture_df())

    def test_quality_reported_for_both_tasks(self):
        q = self.corpus.quality
        self.assertIn("sentiment", q)
        self.assertIn("emotion", q)
        self.assertGreater(q["sentiment"]["accuracy_pct"], 60.0)

    def test_overview_counts(self):
        ov = self.corpus.overview()
        self.assertEqual(ov["n_reviews"], 250)
        self.assertEqual(ov["n_categories"], 2)

    def test_category_stats_ci_brackets_pct(self):
        s = self.corpus.category_stats("Elektronik")
        lo, hi = s["negative_pct_ci95"]
        self.assertLessEqual(lo, s["negative_pct"])
        self.assertLessEqual(s["negative_pct"], hi)
        self.assertIn("anger", s["emotion_pct_all_reviews_PREDICTED"])

    def test_compare_categories_sorted(self):
        rows = self.corpus.compare_categories()
        pcts = [r["negative_pct"] for r in rows]
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_distinctive_terms_surface_planted_term(self):
        out = self.corpus.distinctive_terms("Elektronik", "negative", k=10)
        self.assertIn("baterai", [t["term"] for t in out["terms"]])

    def test_search_finds_reviews(self):
        hits = self.corpus.search("baterai cepat habis", sentiment="negative", n=3)
        self.assertTrue(hits)
        self.assertTrue(all(h["sentiment_pred"] == "negative" for h in hits))

    def test_get_review_redacts_contacts(self):
        rid = self.corpus.df.iloc[0].review_id
        r = self.corpus.get_review(rid)
        self.assertNotIn("081234567890", r["text"])
        self.assertIn("[nomor]", r["text"])

    def test_unknown_category_gives_error_dict(self):
        self.assertIn("error", self.corpus.category_stats("TidakAda"))


if __name__ == "__main__":
    unittest.main()
