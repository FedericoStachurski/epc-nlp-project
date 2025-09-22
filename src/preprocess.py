import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def split_improvements(text: str):
    """Split the 'IMPROVEMENTS' field into individual upgrades."""
    return [imp.strip() for imp in str(text).split('|') if imp.strip()]

def parse_improvement(text: str):
    """Extract measure, cost, saving, and rating from raw improvement text."""
    measure = re.findall(r"Description: (.*?);", text)
    cost = re.findall(r"Indicative Cost: (.*?);", text)
    saving = re.findall(r"Typical Saving: (\d+)", text)
    rating = re.findall(r"Energy Rating after improvement: ([A-G])", text)

    return {
        "measure": measure[0] if measure else None,
        "cost": cost[0] if cost else None,
        "saving": int(saving[0]) if saving else None,
        "rating": rating[0] if rating else None,
        "raw": text
    }

def clean_text(text: str):
    """Basic NLP cleaning: lowercase, remove stopwords/punctuation."""
    text = re.sub(r"[^a-zA-Z ]", " ", str(text).lower())
    return " ".join([w for w in text.split() if w not in STOPWORDS])

def vectorise_text(corpus, max_features=5000):
    """Convert text corpus into TF-IDF features."""
    vectoriser = TfidfVectorizer(max_features=max_features)
    X = vectoriser.fit_transform(corpus)
    return X, vectoriser
