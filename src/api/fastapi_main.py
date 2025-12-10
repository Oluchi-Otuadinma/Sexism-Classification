from fastapi import FastAPI
from pydantic import BaseModel
from src.api.hf_client import hf_predict

app = FastAPI(title="Sexism Detection API")


class TextInput(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input: TextInput):
    result = hf_predict(input.text)
    return result
