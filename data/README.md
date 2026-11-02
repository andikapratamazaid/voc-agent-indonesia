# Data: PRDECT-ID

This project runs on a **real, published dataset** — no synthetic reviews are used for any
headline number.

## The dataset

**PRDECT-ID** (Sutoyo, R., Achmad, S., Chowanda, A., Andangsari, E. W., & Isa, S. M., 2022.
*PRDECT-ID: Indonesian product reviews dataset for emotions classification tasks.* Data in Brief,
44, 108554. DOI: 10.1016/j.dib.2022.108554)

- 5,400 Indonesian product reviews from Tokopedia, across 29 product categories.
- Every review is labelled for **sentiment** (Positive / Negative) and **emotion**
  (Happy, Love, Sad, Fear, Anger), annotated following criteria by clinical-psychology experts.
- Extra columns (used when present): Category, Product Name, Price, Customer Rating.

## The copy in this repo

`PRDECT-ID Dataset.csv` is copied **unmodified** from the dataset's official GitHub repository
(https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset, retrieved
2 Nov 2026). That repository distributes the dataset under the **MIT License** (© Rhio Sutoyo);
this project is not affiliated with the dataset authors. The dataset is also available on Kaggle
and Mendeley Data.

## Format reality (worth knowing)

- The paper describes the file as semicolon-delimited; **the published CSV is actually
  comma-delimited** with quoted text fields, and at least one review contains a line break inside
  its quoted text. `voc/data.py` sniffs the delimiter and matches column names by alias, so both
  the comma and semicolon variants load; it reports the separator and columns it found.
- Label distribution in this copy: sentiment Negative 2,821 / Positive 2,579; emotion Happy 1,770,
  Sadness 1,202, Fear 920, Love 809, Anger 699.

## Using your own comment export instead

Any CSV with a review-text column works. Recognised headers (case-insensitive, Indonesian or
English): `Customer Review` / `Ulasan` / `Review`, plus optional `Sentiment`/`Sentimen`,
`Emotion`/`Emosi`, `Customer Rating`/`Rating`, `Category`/`Kategori`, `Product Name`/`Nama Produk`,
`Price`/`Harga`. Gold labels are only needed for evaluation; the agent labels unlabelled corpora
with the baseline model and says so.
