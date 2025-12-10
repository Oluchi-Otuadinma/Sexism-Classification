import joblib
from src.data.preprocess import clean_text

# Uses the joblib lib to load commonly joint components in ML "trained model" and "Vectoriser"
# Load pre-trained model + vectorizer
model = joblib.load("outputs/models/classifier.joblib")
vectorizer = joblib.load("outputs/models/vectorizer.joblib")


def predict_text(text: str):
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0])

    return {
        "label": str(pred),
        "confidence": float(prob)
    }
