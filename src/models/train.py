import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from src.data.preprocess import clean_text


def train_classifier(csv_path: str, model_out: str):
    df = pd.read_csv(csv_path)
    
    df["clean"] = df["text"].apply(clean_text)

    X = df["clean"]
    y = df["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)

    dump(model, f"{model_out}/classifier.joblib")
    dump(vectorizer, f"{model_out}/vectorizer.joblib")

    print("Model saved!")


if __name__ == "__main__":
    train_classifier("data/processed/train.csv", "outputs/models")
