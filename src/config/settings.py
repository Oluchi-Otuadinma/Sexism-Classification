import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face model name (your fine-tuned model on HF Hub)
HF_MODEL = os.getenv("HF_MODEL", "your-username/sexism-classifier")

# Hugging Face API token
HF_API_KEY = os.getenv("HF_API_KEY")

# Default preprocessing config
PREPROCESS_CONFIG = {
    "lowercase": True,
    "replace_urls": True,
    "extract_domain": False,
    "remove_stopwords": True,
    "lemmatize": True,
}
