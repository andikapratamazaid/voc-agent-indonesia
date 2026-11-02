"""Indonesian marketplace-review normalisation: slang, elongation, price strings, emoji, contact redaction."""
from __future__ import annotations

import re

SLANG = {
    "bgt": "banget", "bngt": "banget", "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak",
    "enggak": "tidak", "engga": "tidak", "tdk": "tidak", "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk",
    "untk": "untuk", "krn": "karena", "krna": "karena", "karna": "karena", "jg": "juga", "sy": "saya", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "sudh": "sudah", "blm": "belum", "belom": "belum", "brg": "barang", "brgnya": "barangnya",
    "cpt": "cepat", "cepet": "cepat", "trims": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih",
    "thx": "terima kasih", "mantul": "mantap", "mantab": "mantap", "mantep": "mantap", "recomended": "rekomendasi",
    "recommended": "rekomendasi", "recommend": "rekomendasi", "ori": "asli", "kcewa": "kecewa", "pke": "pakai", "pake": "pakai",
    "klo": "kalau", "kalo": "kalau", "tp": "tapi", "sm": "sama", "lg": "lagi", "dr": "dari", "dlm": "dalam", "hrs": "harus",
    "bs": "bisa", "bsa": "bisa", "emg": "memang", "bnr": "benar", "bener": "benar", "cm": "cuma", "aja": "saja", "doang": "saja",
    "dpt": "dapat", "dapet": "dapat", "sampe": "sampai", "smpai": "sampai", "krg": "kurang", "msh": "masih", "brp": "berapa",
    "spt": "seperti", "kyk": "seperti", "kayak": "seperti", "bgs": "bagus", "jlk": "jelek", "pengiriman": "pengiriman",
}
EMOJI = {"😡": " marah ", "😠": " marah ", "😍": " suka ", "😭": " sedih ", "😢": " sedih ", "👍": " bagus ", "❤": " suka ",
         "😊": " senang ", "😁": " senang ", "😱": " takut ", "😨": " takut "}
# Function words for display (term lists, themes). Negators are deliberately NOT stopwords.
STOPWORDS = set("""yang dan di ke dari untuk dengan ini itu ada pada juga saya aku kami kita nya sudah akan atau karena sama
saja lagi jadi udah lah deh sih dong ya yah aja pas masih bisa harus mau oleh dalam para atas bagi seperti terima kasih
barang produk toko seller penjual""".split())
NEGATORS = {"tidak", "kurang", "belum", "bukan", "jangan", "tak"}

_PRICE = re.compile(r"(rp\.?\s?\d[\d.,]*|\b\d+(?:[.,]\d+)?\s?(?:rb|ribu|k|jt|juta)\b)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+62|62|0)8\d{8,11}(?!\d)")


def redact(text: str) -> str:
    """Remove e-mail addresses and Indonesian mobile numbers before showing text to a model or a user."""
    return _PHONE.sub("[nomor]", _EMAIL.sub("[email]", str(text)))


def normalize(text) -> str:
    t = str(text if text is not None else "").lower()
    for k, v in EMOJI.items():
        t = t.replace(k, v)
    t = _URL.sub(" ", t)
    t = _PRICE.sub(" harga ", t)
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)                     # bagusss -> bagus (three or more repeats)
    out = []
    for w in t.split():
        out.extend(SLANG.get(w, w).split())
    return " ".join(out)


def tokens_for_display(text) -> list[str]:
    return [w for w in normalize(text).split() if w not in STOPWORDS and len(w) > 1 and not w.isdigit()]
