"""
Centralized configuration for the Sexism Classification project.

This module loads environment variables and provides configuration constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
INFERENCE_DIR = OUTPUTS_DIR / "inference"

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, INFERENCE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
HF_API_KEY: str = os.getenv("HF_API_KEY", "")
HF_MODEL: str = os.getenv("HF_MODEL", "your-model-name")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # For pushing to HuggingFace Hub

# Dataset Configuration
DATASET_PATH: str = os.getenv("DATASET_PATH", str(RAW_DATA_DIR / "dev.csv"))
TEXT_COLUMN: str = os.getenv("TEXT_COLUMN", "text")
LABEL_COLUMN: str = os.getenv("LABEL_COLUMN", "label")

# Model Training Configuration
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE: float = float(os.getenv("VAL_SIZE", "0.1"))

# BERT Model Configuration
BERT_MODEL_NAME: str = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "128"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))
NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))

# Traditional ML Configuration
USE_TFIDF: bool = os.getenv("USE_TFIDF", "True").lower() == "true"
TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "5000"))
TFIDF_NGRAM_RANGE: tuple = (1, 2)  # Unigrams and bigrams

# Preprocessing Configuration
PREPROCESSING_PRESET: str = os.getenv("PREPROCESSING_PRESET", "standard")
MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))

# API Server Configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"

# Inference backend: "local" (transformers, loaded via FastAPI lifespan) or "hf_api" (HuggingFace Inference API)
INFERENCE_BACKEND: str = os.getenv("INFERENCE_BACKEND", "local")
# Pre-trained base model; a fresh classification head is attached on load
MODEL_ID: str = os.getenv("MODEL_ID", "vinai/bertweet-base")
# Fine-tuned checkpoint directory — loaded INSTEAD of MODEL_ID if it contains a saved model
LOCAL_MODEL_DIR: str = os.getenv("LOCAL_MODEL_DIR", str(MODELS_DIR / "bertweet-sexism"))
# Download/load the model at startup (lifespan). Set false to defer to first request.
PRELOAD_MODEL: bool = os.getenv("PRELOAD_MODEL", "True").lower() == "true"

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache Configuration
ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1000"))
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds

# Label Mapping (customize for your dataset)
LABEL_MAP = {
    "not_sexist": 0,
    "sexist": 1
}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Model performance thresholds
MIN_ACCURACY: float = 0.8
MIN_F1_SCORE: float = 0.75


def validate_config() -> bool:
    """
    Validate that required configuration is present.
    
    Returns:
        True if config is valid, raises ValueError otherwise
    """
    errors = []
    
    # Check required API keys
    if not HF_API_KEY and os.getenv("REQUIRE_HF_API_KEY", "False").lower() == "true":
        errors.append("HF_API_KEY is required but not set")
    
    if not HF_MODEL:
        errors.append("HF_MODEL must be specified")
    
    # Check paths exist
    if not PROJECT_ROOT.exists():
        errors.append(f"Project root does not exist: {PROJECT_ROOT}")
    
    # Check numeric ranges
    if not 0 < TEST_SIZE < 1:
        errors.append(f"TEST_SIZE must be between 0 and 1, got {TEST_SIZE}")
    
    if not 0 < VAL_SIZE < 1:
        errors.append(f"VAL_SIZE must be between 0 and 1, got {VAL_SIZE}")
    
    if BATCH_SIZE < 1:
        errors.append(f"BATCH_SIZE must be positive, got {BATCH_SIZE}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def print_config():
    """Print current configuration (excluding sensitive data)."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    print("\n[Paths]")
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Data Directory:   {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory:   {LOGS_DIR}")
    
    print("\n[Dataset]")
    print(f"Dataset Path:     {DATASET_PATH}")
    print(f"Text Column:      {TEXT_COLUMN}")
    print(f"Label Column:     {LABEL_COLUMN}")
    
    print("\n[Training]")
    print(f"Random Seed:      {RANDOM_SEED}")
    print(f"Test Size:        {TEST_SIZE}")
    print(f"Val Size:         {VAL_SIZE}")
    print(f"Batch Size:       {BATCH_SIZE}")
    print(f"Learning Rate:    {LEARNING_RATE}")
    print(f"Epochs:           {NUM_EPOCHS}")
    
    print("\n[Model]")
    print(f"BERT Model:       {BERT_MODEL_NAME}")
    print(f"Max Length:       {MAX_LENGTH}")
    print(f"HF Model:         {HF_MODEL}")
    print(f"HF API Key:       {'*' * 10 if HF_API_KEY else 'Not set'}")
    
    print("\n[API]")
    print(f"Host:             {API_HOST}")
    print(f"Port:             {API_PORT}")
    print(f"Workers:          {API_WORKERS}")
    print(f"Reload:           {API_RELOAD}")
    
    print("\n[Preprocessing]")
    print(f"Preset:           {PREPROCESSING_PRESET}")
    print(f"Min Text Length:  {MIN_TEXT_LENGTH}")
    print(f"Max Text Length:  {MAX_TEXT_LENGTH}")
    
    print("=" * 60)


# NOTE: the .env template lives in .env.example at the project root
