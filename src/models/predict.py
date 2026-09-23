import joblib

from src.config.settings import MODELS_DIR
from src.data.preprocess import clean_text

# Lazy-loaded singletons — the model files are only required when a
# prediction is actually made, so importing this module never crashes
# on a fresh checkout (see TODO history: it used to load at import time).
_model = None
_vectorizer = None


def _load_components():
    """Load the trained classifier + vectorizer on first use."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = MODELS_DIR / "classifier.joblib"
        vectorizer_path = MODELS_DIR / "vectorizer.joblib"
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run training first "
                "(python -m src.models.train), then restart the API."
            )
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vectorizer_path)
    return _model, _vectorizer


def predict_text(text: str):
    model, vectorizer = _load_components()
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0])

    return {
        "label": str(pred),
        "confidence": float(prob),
    }
