# 📰 Fake News Detection

> 7-Class NLP Text Classifier — 87% Accuracy on 12,273 Articles

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154F5B?logo=python&logoColor=white)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Screenshots

### 🤖 Model Training & Results
![Results] (https://raw.githubusercontent.com/serinenadabenmissi-design/machine-learning-fake-news-detection/main/screenshots/fake%20news.png))
### 📊 Classification Output
![Classification] (https://raw.githubusercontent.com/serinenadabenmissi-design/machine-learning-fake-news-detection/master/screenshots/fake%20news%20detection.png)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 87% |
| **Dataset** | 12,273 articles (Kaggle Fake News) |
| **Classes** | 7: bias, conspiracy, fake, hate, junksci, satire, state |
| **Features** | 5,000 TF-IDF features (1-2 n-grams) |
| **Ranking** | Top 15% in university competition |

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core language |
| **scikit-learn** | ML models, evaluation, TF-IDF |
| **NLTK** | Text preprocessing (tokenization, stopwords, lemmatization) |
| **pandas** | Data manipulation and analysis |
| **scipy** | Statistical operations |
| **matplotlib / seaborn** | Visualization |

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip

### Clone Repository

```bash
git clone https://github.com/serinenadabenmissi-design/machine-learning-fake-news-detection.git
cd machine-learning-fake-news-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Train the Model

```bash
python train.py
```

### Predict on New Text

```bash
python predict.py --text "Your news article here"
```

### Jupyter Notebook

```bash
jupyter notebook Fake_News_Detection.ipynb
```

---

## 🧠 Methodology

### Data Pipeline

```
Raw Text
    ↓
NLTK Preprocessing
├── Tokenization
├── Stopword Removal
└── Lemmatization
    ↓
TF-IDF Vectorization
├── 5,000 features
└── N-gram range: 1-2
    ↓
Model Training
├── Logistic Regression
├── Random Forest
├── SVM
└── Naive Bayes
    ↓
Evaluation
├── Stratified K-Fold CV
├── Confusion Matrix
└── Classification Report
```

### Text Preprocessing

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing pipeline
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
```

### Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    preprocessor=preprocess_text
)

X = vectorizer.fit_transform(articles)
```

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 85% | 0.84 | 0.83 | 0.84 |
| **Random Forest** | **87%** | **0.86** | **0.85** | **0.86** |
| SVM | 84% | 0.83 | 0.82 | 0.83 |
| Naive Bayes | 82% | 0.81 | 0.80 | 0.81 |

---

## 📂 Project Structure

```
machine-learning-fake-news-detection/
├── data/
│   └── fake_news_dataset.csv
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── screenshots/
│   ├── fake news detection.png
│   └── fake news.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔮 Future Improvements

- [ ] **Deep Learning Models** — LSTM, BERT for better accuracy
- [ ] **Real-Time API** — Flask/FastAPI endpoint for live predictions
- [ ] **Web Interface** — Streamlit app for interactive classification
- [ ] **Multilingual Support** — French, Arabic fake news detection
- [ ] **Source Credibility Scoring** — Integrate domain reputation

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

**Serine Benmissi**

- 📧 [benmissi.dev@gmail.com](mailto:benmissi.dev@gmail.com)
- 💼 [linkedin.com/in/ben-missi-993269419](https://linkedin.com/in/ben-missi-993269419)
- 🌐 [portfolio-inky-three-33.vercel.app](https://portfolio-inky-three-33.vercel.app)
- 🐱 [github.com/serinenadabenmissi-design](https://github.com/serinenadabenmissi-design)

**⭐ Star this repo if you find it useful!**
