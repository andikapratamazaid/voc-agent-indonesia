import unittest

from tests.helpers import fixture_df
from voc.models import evaluate_holdout, rating_baseline


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture_df()

    def test_sentiment_holdout_beats_majority(self):
        h = evaluate_holdout(self.df, "sentiment")
        self.assertGreater(h["accuracy"], h["majority_baseline_accuracy"])
        self.assertEqual(h["n_train"] + h["n_test"], 250)

    def test_emotion_holdout_structure(self):
        h = evaluate_holdout(self.df, "emotion")
        self.assertEqual(sorted(h["labels"]), ["anger", "fear", "happy", "love", "sadness"])
        self.assertEqual(len(h["confusion"]), 5)
        self.assertTrue(all(len(row) == 5 for row in h["confusion"]))

    def test_rating_baseline_runs(self):
        rb = rating_baseline(self.df)
        self.assertIsNotNone(rb)
        self.assertIn(rb["threshold_negative_if_rating_leq"], (1, 2, 3, 4))
        self.assertGreater(rb["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
