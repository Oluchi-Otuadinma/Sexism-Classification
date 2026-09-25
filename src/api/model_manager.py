"""
Local model manager — loads the transformer for inference via FastAPI's
lifespan (see fastapi_main.lifespan), replacing the lazy get_client()
singleton for local inference.

Loading behaviour (per the project spec):

    Model (AutoModelForSequenceClassification):
        Downloads the pre-trained Transformer weights (vinai/bertweet-base)
        and attaches a fresh classification head (a linear layer) on top that
        outputs probabilities across the target classes.

If a fine-tuned checkpoint exists at ``LOCAL_MODEL_DIR`` it is loaded instead
(inference then uses the trained head rather than a fresh one).
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - environments without torch
    TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL_ID = "vinai/bertweet-base"
LABELS = ["not sexist", "sexist"]


class ModelManager:
    """Owns the model + tokenizer and serves predictions."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        num_labels: int = 2,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
        cache_size: int = 1000,
    ):
        self.model_id = model_id
        self.num_labels = num_labels
        self.checkpoint_dir = checkpoint_dir
        self.device = device or (
            ("cuda" if torch.cuda.is_available() else "cpu") if TRANSFORMERS_AVAILABLE else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.used_fresh_head = False
        self.labels = LABELS[:num_labels] if len(LABELS) >= num_labels else [
            f"class_{i}" for i in range(num_labels)
        ]
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ load
    def load(self) -> "ModelManager":
        """Load the model into memory. Heavy — called from the lifespan handler."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers/torch are not installed — local inference unavailable. "
                "Set INFERENCE_BACKEND=hf_api to use the HuggingFace Inference API instead."
            )

        start = time.time()
        checkpoint_ready = bool(
            self.checkpoint_dir and os.path.exists(os.path.join(self.checkpoint_dir, "config.json"))
        )

        if checkpoint_ready:
            logger.info(f"Loading fine-tuned checkpoint from {self.checkpoint_dir} ...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
            self.used_fresh_head = False
        else:
            logger.info(
                f"Model (AutoModelForSequenceClassification): downloading pre-trained "
                f"Transformer weights ({self.model_id}) and attaching a fresh "
                f"classification head (linear layer, {self.num_labels} classes)..."
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, num_labels=self.num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.used_fresh_head = True
            logger.warning(
                "Fresh (randomly initialised) classification head — predictions are "
                "untrained until the model is fine-tuned and a checkpoint is saved "
                "to LOCAL_MODEL_DIR."
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info(
            f"Model ready on '{self.device}' in {time.time() - start:.1f}s "
            f"({'fresh head' if self.used_fresh_head else 'fine-tuned checkpoint'})"
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    # --------------------------------------------------------------- predict
    def predict(self, text: str) -> Dict:
        """Classify one text; returns label, confidence, full probabilities."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return {**self._cache[key], "cached": True}

        self._ensure_loaded()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        best = int(probs.argmax())
        result = {
            "label": self.labels[best],
            "confidence": float(probs[best]),
            "probabilities": {label: float(p) for label, p in zip(self.labels, probs)},
        }

        self._cache[key] = result
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return {**result, "cached": False}

    # ----------------------------------------------------------------- cache
    def get_cache_stats(self) -> Dict:
        hits = getattr(self, "_hits", 0)
        misses = getattr(self, "_misses", 0)
        return {
            "hits": hits,
            "misses": misses,
            "size": len(self._cache),
            "maxsize": self._cache_size,
            "hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else 0.0,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    # -------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        """Release model memory (called on lifespan shutdown)."""
        if self.model is not None:
            logger.info("Releasing model memory...")
            del self.model, self.tokenizer
            self.model = self.tokenizer = None
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def info(self) -> Dict:
        return {
            "backend": "local",
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "fresh_head": self.used_fresh_head,
            "labels": self.labels,
        }
