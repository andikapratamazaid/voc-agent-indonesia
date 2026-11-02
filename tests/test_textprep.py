import unittest

from voc.textprep import normalize, redact, tokens_for_display


class TestTextPrep(unittest.TestCase):
    def test_slang_expanded(self):
        out = normalize("barangnya gak bagus bgt")
        self.assertIn("tidak", out)
        self.assertIn("banget", out)

    def test_elongation_collapsed(self):
        self.assertIn("bagus", normalize("bagusss sekali"))

    def test_emoji_becomes_word(self):
        self.assertIn("marah", normalize("seller tidak respon 😡"))

    def test_price_becomes_token(self):
        out = normalize("harga Rp50rb kemahalan")
        self.assertIn("harga", out)
        self.assertNotIn("50rb", out)

    def test_redact_contacts(self):
        out = redact("hubungi 081234567890 atau a.b@mail.com ya")
        self.assertIn("[nomor]", out)
        self.assertIn("[email]", out)
        self.assertNotIn("081234567890", out)

    def test_display_tokens_keep_negator_drop_stopwords(self):
        toks = tokens_for_display("barang ini tidak bagus dan sangat lambat")
        self.assertIn("tidak", toks)
        self.assertNotIn("dan", toks)
        self.assertNotIn("barang", toks)  # domain stopword by design

    def test_normalize_handles_none(self):
        self.assertEqual(normalize(None), "")


if __name__ == "__main__":
    unittest.main()
