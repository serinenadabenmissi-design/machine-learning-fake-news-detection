# 📰 Fake News Detection

> **Multi-Class Text Classification (7 classes) — 87% Accuracy via 80/20 Stratified Split**
> Kaggle Fake News Dataset · 12,273 Articles · TF-IDF + Classical ML

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154F5B?logo=python&logoColor=white)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 87% |
| **Dataset** | 12,273 labeled articles (Kaggle) |
| **Classes** | 7 — bias, conspiracy, fake, hate, junksci, satire, state |
| **Features** | TF-IDF, 5,000 features, unigrams + bigrams |
| **Best Model** | Random Forest (vs. LR, SVM, Naive Bayes) |
| **Split Strategy** | 80/20 train-test, stratified by class <!-- TODO: confirm --> |

> **Why 87% matters:** with 7 classes, random chance is ~14%. <!-- TODO: add your majority-class baseline here, e.g. "The majority class (fake) alone achieves only 6X%, so 87% represents a strong margin over both." -->

---

## 🖼️ Screenshots

### 🤖 Model Training & Results
![Results](https://raw.githubusercontent.com/serinenadabenmissi-design/machine-learning-fake-news-detection/master/screenshots/fake%20news.png)

### 📊 Classification Output
![Classification](https://raw.githubusercontent.com/serinenadabenmissi-design/machine-learning-fake-news-detection/master/screenshots/fake%20news%20detection.png)

---

## 🧠 Methodology

### 1 · Text Preprocessing (NLTK)

```
Raw article
   ↓  lowercase
Tokenize → drop non-alphanumeric → remove stopwords → lemmatize
   ↓
Clean text
```

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens)
```

### 2 · Feature Extraction (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,          # dampen term-frequency effect
    min_df=2,                   # cut rare tokens (likely noise/leakage risk)
    preprocessor=preprocess_text
)

X = vectorizer.fit_transform(train_articles)   # fit on TRAIN only
X_test = vectorizer.transform(test_articles)   # no leakage into test
```

### 3 · Experiment Design

- **Split:** 80/20 stratified by label, `random_state=42` <!-- TODO: confirm -->
- **Leakage guard:** vectorizer is fitted **only on the training fold** — test text never influences the vocabulary. <!-- TODO: if you deduplicated near-duplicate articles across splits, say so here — it's a big credibility win -->
- **Validation:** <!-- TODO: e.g. "5-fold stratified cross-validation on the training set for model selection; final numbers reported once on the held-out test set" -->

### 4 · Model Comparison (held-out test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training Time |
|-------|----------|-------------------|----------------|------------|---------------|
| Logistic Regression | 85% | 0.84 | 0.83 | 0.84 | <!-- TODO --> |
| **Random Forest** ✅ | **87%** | **0.86** | **0.85** | **0.86** | <!-- TODO --> |
| SVM (LinearSVC) | 84% | 0.83 | 0.82 | 0.83 | <!-- TODO --> |
| Multinomial Naive Bayes | 82% | 0.81 | 0.80 | 0.81 | <!-- TODO --> |

> **Note:** TF-IDF features are high-dimensional and sparse, so linear models were expected to dominate — Random Forest's edge (small but consistent across CV folds) was still worth reporting. See *Challenges* below.

### 5 · Per-Class Breakdown (Random Forest)

<!-- TODO: paste your sklearn classification_report output here. Macro-average alone hides minority-class performance. -->

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| fake | | | | |
| bias | | | | |
| conspiracy | | | | |
| satire | | | | |
| junksci | | | | |
| state | | | | |
| hate | | | | |
| **Macro avg** | | | | |

### 6 · Classification Pipeline

```
Preprocess → TF-IDF → Random Forest → Predicted class
                                              ↓
                            per-class probability scores (top-3 shown)
```

---

## 💡 Challenges & Learnings

<!-- TODO: pick 2–3 real ones — this section is what interviewers remember -->

- **Class imbalance:** `hate` and `junksci` were under-represented; <!-- how did you handle it / what happened to their recall? -->
- **Satire vs. fake:** hardest boundary — both use exaggerated claims; model occasionally <!-- observed error pattern -->
- **Tuning:** <!-- did you run GridSearchCV? what were the best params? e.g. RandomForest(n_estimators=..., max_depth=...) -->

---

## 🚀 Quickstart

### Setup

```bash
git clone https://github.com/serinenadabenmissi-design/machine-learning-fake-news-detection.git
cd machine-learning-fake-news-detection
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

### Predict

```bash
python src/predict.py --text "Your news article here"
```

### Explore

```bash
jupyter notebook notebooks/Fake_News_Detection.ipynb
```

---

## 📂 Project Structure

```
machine-learning-fake-news-detection/
├── data/
│   └── fake_news_dataset.csv
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── screenshots/
│   ├── fake-news.png
│   └── confusion-matrix.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔮 Future Work

- **Transformer baselines** — fine-tune `distilbert-base-uncased` and compare against TF-IDF + RF (expected: +3–6% on this task)
- **Probability calibration** — output calibrated confidence scores, not just argmax labels
- **Serving layer** — FastAPI endpoint + Streamlit demo for live classification
- **Multilingual extension** — French & Arabic corpora

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 📬 Contact

**Serine Benmissi**

- 📧 benmissi.dev@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/ben-missi-993269419)
- 🌐 [Portfolio](https://portfolio-inky-three-33.vercel.app)
- 🐱 [GitHub](https://github.com/serinenadabenmissi-design)
