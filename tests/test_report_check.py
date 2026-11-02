import unittest

from voc.report_check import check_report

TEXTS = {"R00001": "pengiriman lambat sekali barang rusak parah",
         "R00002": "barang bagus banget puas sekali"}
ALLOWED = {100.0, 250.0, 60.0, 40.0}


class TestReportCheck(unittest.TestCase):
    def test_clean_report_passes(self):
        rep = ('Dari 250 ulasan, 60.0% diprediksi negatif. Emosi marah adalah prediksi model. '
               'Contoh: "pengiriman lambat sekali" [R00001].')
        res = check_report(rep, TEXTS, ALLOWED)
        self.assertTrue(res.passed, res)

    def test_fake_id_flagged(self):
        res = check_report('Lihat "pengiriman lambat sekali" [R99999].', TEXTS, ALLOWED)
        self.assertEqual(res.invalid_ids, ["R99999"])
        self.assertFalse(res.passed)

    def test_invented_percentage_flagged(self):
        res = check_report("Sebanyak 87,3% pelanggan kecewa.", TEXTS, ALLOWED)
        self.assertIn("87,3%", res.unsupported_numbers)

    def test_altered_quote_unverified(self):
        res = check_report('Contoh: "pengiriman sangat lambat sekali" [R00001].', TEXTS, ALLOWED)
        self.assertEqual(len(res.unverified_quotes), 1)

    def test_long_quote_without_citation_flagged(self):
        res = check_report('Kata pelanggan "pengiriman lambat sekali barang rusak parah" begitu saja.', TEXTS, ALLOWED)
        self.assertEqual(len(res.uncited_quotes), 1)

    def test_emotion_percentage_without_caveat_flagged(self):
        res = check_report("Emosi marah mencapai 40.0% dari ulasan.", TEXTS, ALLOWED)
        self.assertTrue(res.emotion_without_caveat)

    def test_emotion_percentage_with_caveat_ok(self):
        res = check_report("Emosi marah 40.0% (prediksi model, akurasi terbatas).", TEXTS, ALLOWED)
        self.assertFalse(res.emotion_without_caveat)


if __name__ == "__main__":
    unittest.main()
