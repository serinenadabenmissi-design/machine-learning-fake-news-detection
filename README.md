# 📰 Fake News Detection

> **Two-Stage Text Classification (8 classes) — Random Forest + TF-IDF**
> 12,273 articles · Kaggle Fake News Dataset

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🖼️ Screenshots

### 🤖 Model Training & Results
![Training results](screenshots/fake-news.png)

### 📊 Confusion Matrix & Per-Class Performance
![Confusion matrix](screenshots/confusion-matrix.png)

## Overview

This project classifies news articles into 8 categories (`bs`, `bias`, `conspiracy`, `fake`, `hate`, `junksci`, `satire`, `state`) using a **two-stage Random Forest pipeline**:

1. **Stage 1 — Binary filter:** is this article `bs` or `others`?
2. **Stage 2 — Fine-grained classifier:** for anything flagged `others`, which of the 7 specific categories is it?

The two-stage design exists because the dataset is heavily imbalanced — `bs` alone makes up **89%** of all articles — so splitting the problem this way keeps the fine-grained classifier from being drowned out by the majority class.

## Dataset

| | |
|---|---|
| Raw articles | 12,999 |
| After cleaning (dropped rows missing title/text) | 12,273 |
| Classes | 8 — `bs`, `bias`, `conspiracy`, `fake`, `hate`, `junksci`, `satire`, `state` |
| Class balance | `bs`: 89% · everything else combined: 11% |
| Features | TF-IDF on title + text (5,000 features) · one-hot `country` · `domain_rank` (normalized) |
| Split | 70/30 train/test, stratified by class, `random_state=42` |

## Results

### Stage 1 — Binary (`bs` vs `others`)

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bs | 0.92 | 1.00 | 0.96 | 3,276 |
| others | 0.92 | 0.30 | 0.45 | 406 |
| **Accuracy** | | **0.92** | | 3,682 |

### Stage 2 — Fine-grained (7 classes, evaluated on the true `others` subset)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bias | 0.86 | 0.91 | 0.88 | 106 |
| conspiracy | 0.79 | 0.86 | 0.82 | 124 |
| fake | 0.00 | 0.00 | 0.00 | 6 |
| hate | 0.81 | 0.77 | 0.79 | 74 |
| junksci | 1.00 | 0.93 | 0.97 | 30 |
| satire | 0.96 | 0.90 | 0.93 | 30 |
| state | 1.00 | 0.89 | 0.94 | 36 |
| **Accuracy** | | **0.85** | | 406 |

## ⚠️ Architecture & Known Limitations

Being upfront about this because it's the most important thing to understand about the system's real-world behavior:

- **`bs` is 89% of the dataset**, so Stage 1's 92% accuracy is only a few points above what you'd get by always guessing `bs`. The real signal is in the per-class numbers, not the headline accuracy.
- **Stage 1's recall on `others` is 0.30** — meaning roughly 70% of true non-`bs` articles are misclassified as `bs` and never reach Stage 2 at all. Stage 2's 85% accuracy is measured on articles *already known* to be non-`bs` — it does not reflect true end-to-end accuracy in a live pipeline, which would be meaningfully lower once Stage 1's filtering losses are accounted for.
- **`fake` (6 test samples) is effectively unlearnable at this sample size** — 19 examples exist in the *entire* 12,273-article dataset. The 0.00 score isn't a bug, it's a data volume problem.
- **Class imbalance is the core challenge of this dataset**, not model choice — a next step worth pursuing is oversampling minority classes (e.g. SMOTE on the TF-IDF space) or collecting more `fake`/`junksci`/`state` examples before further model tuning.

## Reproducing the results

```bash
git clone https://github.com/serinenadabenmissi-design/machine-learning-fake-news-detection.git
cd machine-learning-fake-news-detection
pip install -r requirements.txt
jupyter notebook final.ipynb
```

To measure true end-to-end pipeline accuracy (chaining Stage 1's actual predictions into Stage 2, rather than evaluating Stage 2 on ground-truth `others` labels), add this after training both models:

```python
pred_others_mask = (y_pred_binary == "others")
final_preds = np.array(y_pred_binary, dtype=object)

X_test_pred_others = X_test_normalized[pred_others_mask]
final_preds[pred_others_mask] = multi_model.predict(X_test_pred_others)

print("End-to-end accuracy:", accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds, zero_division=0))
```

## Pipeline

```
Article (title + text) ──┐
                          ├─► TF-IDF (5,000 feat.) ─┐
Country ─────────────────┤                          ├─► Normalizer ─► Stage 1: bs vs others
Domain rank ──────────────┘                          │                        │
                                                       │                  (if "others")
                                                       └─────────────────────► Stage 2: 7-class RF
```

## Project Structure

```
machine-learning-fake-news-detection/
├── data/
│   └── fake.csv
├── final.ipynb
├── preprocessor.pkl
├── normalizer.pkl
├── binary_model.pkl
├── multi_model.pkl
├── requirements.txt
├── README.md
└── LICENSE
```

## Future Work

- Address Stage 1's minority-class recall (class weighting is already applied — try SMOTE, or a lower decision threshold for `others`, or collecting more data)
- Evaluate the true end-to-end pipeline accuracy (see snippet above) and report it alongside the per-stage numbers
- Try a linear model (Logistic Regression / LinearSVC) as a baseline comparison against Random Forest on this TF-IDF feature space — high-dimensional sparse text features often favor linear models
- Transformer baseline (`distilbert-base-uncased`) for comparison

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Serine Benmissi**

- 📧 benmissi.dev@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/ben-missi-993269419)
- 🌐 [Portfolio](https://portfolio-inky-three-33.vercel.app)
- 🐱 [GitHub](https://github.com/serinenadabenmissi-design)
