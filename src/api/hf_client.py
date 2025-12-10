import requests
from src.config.settings import HF_API_KEY, HF_MODEL

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def hf_predict(text: str):
    payload = {"inputs": text}

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    result = response.json()[0]
    label = result["label"]
    score = result["score"]

    return {
        "label": label,
        "confidence": score
    }
