import tempfile
import unittest
from pathlib import Path

from voc.data import load_reviews


def _write(text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


class TestLoader(unittest.TestCase):
    def test_semicolon_indonesian_headers(self):
        path = _write("Ulasan;Sentimen;Emosi;Rating;Kategori\n"
                      "barang bagus banget;Positif;Senang;5;Elektronik\n"
                      "kecewa parah;Negatif;Marah;1;Elektronik\n")
        df, info = load_reviews(path)
        self.assertEqual(info["separator"], ";")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0].sentiment, "positive")
        self.assertEqual(df.iloc[0].emotion, "happy")
        self.assertEqual(df.iloc[1].emotion, "anger")
        self.assertEqual(df.iloc[0].review_id, "R00001")

    def test_comma_prdect_style_headers(self):
        path = _write('Category,Product Name,Customer Rating,Customer Review,Sentiment,Emotion\n'
                      'Kitchen,Wajan,5,"bagus, awet",Positive,Happy\n')
        df, info = load_reviews(path)
        self.assertEqual(info["separator"], ",")
        self.assertEqual(df.iloc[0].category, "Kitchen")
        self.assertEqual(df.iloc[0].rating, 5.0)
        self.assertEqual(df.iloc[0].text, "bagus, awet")

    def test_unmapped_labels_counted(self):
        path = _write("Ulasan;Sentimen;Emosi\nbagus;Netral;Biasa\n")
        df, info = load_reviews(path)
        self.assertEqual(info["unmapped_sentiment"], 1)
        self.assertEqual(info["unmapped_emotion"], 1)

    def test_missing_text_column_raises(self):
        path = _write("Sesuatu;Apa\nx;y\n")
        with self.assertRaises(ValueError):
            load_reviews(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_reviews(str(Path(tempfile.gettempdir()) / "tidak-ada-sama-sekali.csv"))

    def test_empty_reviews_dropped(self):
        path = _write("Ulasan;Sentimen\n;Positif\nisi;Positif\n")
        df, info = load_reviews(path)
        self.assertEqual(info["rows_read"], 2)
        self.assertEqual(info["rows_kept"], 1)


if __name__ == "__main__":
    unittest.main()
