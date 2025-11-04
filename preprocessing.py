# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path, names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

def preprocess(df, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
