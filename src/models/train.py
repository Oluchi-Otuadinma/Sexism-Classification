"""
Train the TF-IDF + Logistic Regression baseline classifier.

Usage:
    python -m src.models.train

Expects a processed dataset at data/processed/train.csv with columns
'text' and 'label' (configurable via settings). Produces:
    outputs/models/classifier.joblib
    outputs/models/vectorizer.joblib
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from joblib import dump

from src.config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.data.preprocess import clean_text


def train_classifier(
    csv_path: str,
    model_out: str,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
):
    df = pd.read_csv(csv_path)

    df["clean"] = df[text_column].apply(clean_text)

    X = df["clean"]
    y = df[label_column]

    # Held-out test split so reported metrics aren't on training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_train_vec, y_train)

    # Evaluate on the held-out split
    preds = model.predict(X_test_vec)
    print("Held-out evaluation:")
    print(classification_report(y_test, preds))
    print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")

    dump(model, f"{model_out}/classifier.joblib")
    dump(vectorizer, f"{model_out}/vectorizer.joblib")

    print("Model saved!")


if __name__ == "__main__":
    train_classifier(
        str(PROCESSED_DATA_DIR / "train.csv"),
        str(MODELS_DIR),
    )
