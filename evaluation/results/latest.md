# Evaluation on the real PRDECT-ID dataset

- Reviews: 5400 | categories: 29 | sentiment gold: {'negative': 2821, 'positive': 2579} | emotion gold: {'happy': 1770, 'sadness': 1202, 'fear': 920, 'love': 809, 'anger': 699}
- Split: stratified holdout 80/20, seed 42. Model: word+char TF-IDF + logistic regression (the project baseline, not a transformer).

### Sentiment (holdout, n_train=4320, n_test=1080)

- Accuracy **95.3%**, macro-F1 **95.3%** (majority-class baseline accuracy 52.2%)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.94 | 0.98 | 0.96 | 564 |
| positive | 0.97 | 0.93 | 0.95 | 516 |

Confusion (rows = gold, columns = predicted, order: negative, positive):

```
  550     14
   37    479
```

### Emotion (holdout, n_train=4320, n_test=1080)

- Accuracy **65.6%**, macro-F1 **62.8%** (majority-class baseline accuracy 32.8%)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.49 | 0.53 | 0.51 | 140 |
| fear | 0.47 | 0.50 | 0.48 | 184 |
| happy | 0.85 | 0.78 | 0.81 | 354 |
| love | 0.68 | 0.74 | 0.71 | 162 |
| sadness | 0.63 | 0.61 | 0.62 | 240 |

Confusion (rows = gold, columns = predicted, order: anger, fear, happy, love, sadness):

```
   74     37      2      0     27
   38     92      6      2     46
    6      8    277     52     11
    1      2     39    120      0
   32     57      3      2    146
```

### Rating-only baseline (sentiment)

- Rule 'negative if rating <= 3': accuracy **98.4%**, macro-F1 98.4% (n_test=1080). The text model has to beat this to be worth anything.

- On PRDECT-ID the rating rule WINS (98.4% vs the text model's 95.3%): in this dataset the star rating and the sentiment label move together almost perfectly. The text model still earns its place where it is actually deployed: social-media and CX comment streams carry no star rating at all, and only the text model produces the five emotion labels. Treat the rating result as a property of this dataset, not as a reason to skip text analysis.

### Corpus out-of-fold label quality (what the agent's tools report)

- sentiment: accuracy 96.0%, macro-F1 96.0% (5-fold out-of-fold, n=5400)
- emotion: accuracy 65.2%, macro-F1 62.1% (5-fold out-of-fold, n=5400)

### Report-audit self-test

- Deterministic demo report: passed=True, issues=0 (must pass).
- Tampered report (invented 87,3%, fake ID R99999, fabricated quote): passed=False, issues=4 (invalid_ids=['R99999'], unsupported_numbers=['87,3%'], unverified_quotes=1, uncited_quotes=1; must fail).

> Emotion is a five-way judgement call; published transformer results on this dataset reach roughly 75-80%, so a TF-IDF baseline below that is expected, not a bug. Emotion shares from this agent are always labelled as model predictions.
