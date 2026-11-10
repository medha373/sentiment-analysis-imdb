# Sentiment Analysis Tool - IMDB Movie Reviews

A machine learning model that classifies movie reviews as **positive** or **negative** using NLP.

## Overview
- **Dataset:** [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25,000 train / 25,000 test reviews, balanced)
- **Text preprocessing:** HTML tag removal, lowercasing, punctuation/number removal
- **Features:** TF-IDF with unigrams and bigrams (top 100,000 features)
- **Model:** Logistic Regression
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Project Structure
```
sentiment-analysis-imdb/
├── train.py            # Load data, train, evaluate, save model
├── predict.py          # Predict sentiment for custom text
├── requirements.txt
├── outputs/
│   ├── metrics.json
│   └── confusion_matrix.png
└── README.md
```

## How to Run
```bash
git clone https://github.com/<your-username>/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
pip install -r requirements.txt
python train.py
python predict.py "This movie was absolutely fantastic!"
```

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | XX.XX% |
| Precision | XX.XX% |
| Recall    | XX.XX% |
| F1-score  | XX.XX% |

![Confusion Matrix](outputs/confusion_matrix.png)

## Example Predictions
```
Positive (97%) -> This movie was absolutely fantastic, I loved every minute of it!
Negative (98%) -> Terrible plot, bad acting, and a complete waste of time.
```

## Key Learnings
- Bigrams capture negation and phrases ("not good", "waste of time") that single words miss.
- Stop words were kept on purpose, since words like "not" carry sentiment.
- A linear model with TF-IDF is a strong, fast baseline for text classification.

## Future Improvements
- Add neutral class with a 3-class dataset
- Try transformer models (e.g. DistilBERT)
- Build a Streamlit web app for live predictions

## Tech Stack
Python, scikit-learn, Hugging Face Datasets, Matplotlib, Joblib
