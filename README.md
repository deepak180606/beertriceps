# Spam Detection System

### NLP-powered SMS spam classifier with real-time confidence scoring

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-22C55E?style=for-the-badge)

![Project banner](assets/spam-app.png)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Performance](#performance)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Running in Google Colab](#running-in-google-colab)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Contributing](#contributing)
15. [License](#license)
16. [Authors](#authors)

---

## Overview

This project classifies SMS/text messages as **spam** or **ham** (legitimate, i.e. a normal message the user actually wants). It combines classic NLP preprocessing, TF-IDF feature extraction and a small neural network built with TensorFlow/Keras, and presents the result in a Streamlit web app together with a confidence percentage.

Spam messages include scams, fake prizes, phishing links and unwanted advertising. Filtering them protects users from fraud and saves time.

---

## Features

- **Text cleaning**: lowercasing, removal of non-letter characters, stop-word removal and Porter stemming
- **TF-IDF vectorization** with the 3,000 most informative terms
- **Neural network classifier** (Keras) with a sigmoid output
- **Confidence scoring**: the app shows the probability behind each decision
- **Interactive Streamlit dashboard** with a message box, example buttons and a result card
- **Colab notebook workflow** for training, evaluation and live predictions

---

## How It Works

```
Input text  ->  Cleaning  ->  TF-IDF (3000 features)  ->  Neural network  ->  Spam probability
```

1. **Cleaning**: the text is lowercased, everything except letters and spaces is removed, English stop words are dropped and each word is reduced to its stem (for example "claiming" becomes "claim").
2. **Vectorization**: the cleaned text is converted into a 3,000-value TF-IDF vector.
3. **Prediction**: the neural network outputs a probability between 0 and 1.
4. **Decision**: a probability above 0.5 is labelled **SPAM**; otherwise the message is labelled **HAM**. The confidence shown to the user is the probability of the predicted class.

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| NLP | NLTK (stop words, Porter stemmer) |
| Feature extraction | scikit-learn (`TfidfVectorizer`) |
| Data handling | pandas, NumPy, joblib |
| Visualization | matplotlib, seaborn |
| Web interface | Streamlit |

---

## Project Structure

```
spam-detection-system/
├── app.py                  # Streamlit web application
├── spam_classifier.keras   # Trained Keras model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer (3000 features)
├── requirements.txt        # Python dependencies
├── assets/
│   └── spam-app.png        # App banner / screenshot
├── README.md
└── LICENSE
```

---

## Dataset

The model is trained on the public **SMS Spam Collection** dataset (UCI Machine Learning Repository).

| Item | Value |
|---|---|
| Total messages | 5,572 |
| Ham (legitimate) | 4,825 (86.59%) |
| Spam | 747 (13.41%) |
| Missing values | 0 |
| Duplicate messages | 403 |
| Average length (ham / spam) | 71.5 / 138.7 characters |

Spam messages are on average about twice as long as ham messages. The dataset is imbalanced, so the train/test split is stratified to keep the same spam share (about 13.4%) in both parts.

- Train/test split: 80% / 20% (4,457 / 1,115 messages), `random_state = 42`

---

## Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3000,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

| Layer | Output shape | Parameters |
|---|---|---|
| Dense (ReLU) | 128 | 384,128 |
| Dropout (0.5) | 128 | 0 |
| Dense (ReLU) | 64 | 8,256 |
| Dense (sigmoid) | 1 | 65 |
| **Total trainable** | | **392,449** |

Training settings: Adam optimizer, binary cross-entropy loss, 10 epochs, batch size 32, 10% of the training data used for validation.

---

## Performance

Results on the held-out test set (1,115 messages):

| Metric | Score |
|---|---|
| Accuracy | 98.03% |
| Precision (spam) | 95.68% |
| Recall (spam) | 89.26% |
| F1-score (spam) | 92.36% |

Confusion matrix:

| | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 960 | 6 |
| **Actual spam** | 16 | 133 |

Of 149 spam messages, 133 were caught and 16 were missed; only 6 of 966 legitimate messages were wrongly flagged as spam. Exact values can vary slightly between runs and TensorFlow versions.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd spam-detection-system

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contains: `streamlit`, `tensorflow`, `scikit-learn`, `nltk`, `joblib`, `numpy`.

---

## Usage

### Run the web app

```bash
streamlit run app.py
```

Then:

1. Paste or type a message in the text box.
2. Click **Analyze Now**.
3. Read the result card: **SPAM DETECTED** or **LEGITIMATE MESSAGE**, with the probability.
4. Use the example buttons to try a normal message, a prize scam or a phishing link.

### Use the model in Python

```python
import joblib, tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = tf.keras.models.load_model("spam_classifier.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")
ps, stop_words = PorterStemmer(), set(stopwords.words("english"))

def clean_text(text):
    text = "".join(c for c in text.lower() if c.isalpha() or c == " ")
    return " ".join(ps.stem(w) for w in text.split() if w not in stop_words)

def predict_spam(text):
    vec = tfidf.transform([clean_text(text)]).toarray()
    return float(model.predict(vec, verbose=0)[0][0])

print(predict_spam("You've won a free prize! Click here to claim!"))
```

---

## Running in Google Colab

The full pipeline (data loading, exploration, cleaning, TF-IDF, training, evaluation, saving and live predictions) can be run cell by cell in Google Colab. The notebook downloads the dataset automatically and ends with an interactive cell where you type your own messages and receive a SPAM/HAM result with its probability.

---

## Limitations

- **Not perfect on phishing links**: a short delivery-scam message containing only a link (for example "Your package is waiting for delivery. Please confirm your address at: http://bit.ly/malicious-link") was classified as HAM with 74.57% confidence.
- **Spam recall is 89%**: about 1 in 10 spam messages in the test set was missed.
- **Dataset age and language**: the SMS Spam Collection is a small, older English dataset, so newer spam styles and other languages may not be detected well.
- **Bag-of-words limits**: TF-IDF ignores word order and context.
- **Missing asset**: `app.py` displays `assets/notspam.png` for legitimate results, but this image is not included in the repository, so that result view will raise an error until the file is added (or the line is removed).

---

## Future Work

- Add URL and phone-number features to improve phishing detection
- Try sequence models (LSTM) or transformer embeddings such as BERT
- Train on a larger, more recent and multilingual dataset
- Handle class imbalance with class weights or resampling
- Deploy the app online (Streamlit Community Cloud)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes and push the branch
4. Open a Pull Request

---

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

## Authors

- **[Your Name]** – [URN / Roll No.]
- **[Teammate Name]** – [URN / Roll No.]

Based on the open-source Spam Detection System by Au Amores.
