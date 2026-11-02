"""Deterministic labelled fixture corpus for tests (NOT evaluation data).

Templated Indonesian marketplace reviews with known labels, so model and corpus behaviour can be
tested without the real dataset. The real evaluation lives in evaluation/run_eval.py and runs on
PRDECT-ID itself.
"""
from __future__ import annotations

import random

import pandas as pd

_POS_HAPPY = [
    "barang bagus banget kualitas oke pengiriman cepat puas sekali",
    "produk mantap sesuai deskripsi packing rapi senang sekali belanja di sini",
    "kualitas bagus harga terjangkau pengiriman cepat recommended banget",
    "puas banget barang asli berfungsi normal seller ramah fast respon",
]
_POS_LOVE = [
    "suka banget sama produk ini jatuh cinta deh bakal langganan terus",
    "love banget hasilnya bagus sekali sesuai ekspektasi terima kasih seller",
    "cinta mati sama kualitasnya bagus banget tidak nyesel beli di sini",
    "sukaa banget warnanya cantik bahannya adem love pokoknya",
]
_NEG_SAD = [
    "kecewa barang tidak sesuai deskripsi sedih sekali rasanya",
    "sedih banget barang datang kotor dan lecet tidak seperti foto",
    "kecewa berat kualitas jauh dari harapan menyesal beli",
    "produk mengecewakan sedih uang terbuang sia sia",
]
_NEG_ANGER = [
    "marah banget penjual tidak respon barang rusak parah kapok",
    "parah banget barang pecah seller cuek emosi sekali komplain tidak digubris",
    "sangat marah pengiriman lama barang cacat minta refund susah",
    "jengkel sekali barang salah kirim penjual lepas tangan marah",
]
_NEG_FEAR = [
    "takut barang ini palsu khawatir tidak awet was was pakainya",
    "khawatir kualitasnya abal abal takut cepat rusak cemas sekali",
    "was was belanja di sini takut penipuan barang tidak jelas asal usulnya",
    "takut kecewa lagi trauma pernah ketipu cemas barang tidak ori",
]
_ELEC_NEG_EXTRA = " baterai cepat habis dan panas"
_CATS = ["Elektronik", "Kecantikan"]


def fixture_df(n_per_emotion: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    groups = [(_POS_HAPPY, "positive", "happy", (4, 5)), (_POS_LOVE, "positive", "love", (5, 5)),
              (_NEG_SAD, "negative", "sadness", (1, 2)), (_NEG_ANGER, "negative", "anger", (1, 1)),
              (_NEG_FEAR, "negative", "fear", (2, 3))]
    rows = []
    for templates, sentiment, emotion, (rlo, rhi) in groups:
        for i in range(n_per_emotion):
            cat = _CATS[i % 2]
            text = rng.choice(templates)
            if sentiment == "negative" and cat == "Elektronik" and i % 2 == 0:
                text += _ELEC_NEG_EXTRA
            rows.append({"text": text, "sentiment": sentiment, "emotion": emotion,
                         "rating": float(rng.randint(rlo, rhi)), "category": cat, "product": "produk uji"})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "review_id", [f"R{i + 1:05d}" for i in range(len(df))])
    df.loc[0, "text"] = df.loc[0, "text"] + " hubungi 081234567890 atau email saya di uji@mail.com"
    return df
