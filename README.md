# Sexism-Classification
Sexism Classification API service (Hugging Face inference API) exposed through FastAPI
## Project Structure

```text
project-name/
│
├── data/
│   ├── raw/           # Original datasets (e.g., Kaggle sexism dataset)
│   ├── processed/     # Cleaned CSV, tokenized text, train/test splits
│   └── external/      # Any external datasets or embeddings
│
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA in Colab
│   ├── 02_preprocessing.ipynb  # Cleaning, balancing, label encoding
│   ├── 03_modeling.ipynb       # Train LogReg/SVM/CNN/BERT
│   ├── 04_evaluation.ipynb     # Metrics, confusion matrix
│   └── 05_export_model.ipynb   # Convert to HF format + push to hub
│
├── src/
│   ├── data/
│   │   ├── load_data.py       # Data loading utilities
│   │   └── preprocess.py      # Text cleaning, tokenization
│   │
│   ├── models/
│   │   ├── train.py           # Training utilities
│   │   ├── predict.py         # Prediction wrapper
│   │   └── utils.py           # Shared NLP helpers
│   │
│   ├── api/
│   │   ├── fastapi_main.py    # Your FastAPI backend
│   │   └── hf_client.py       # HuggingFace API wrapper
│   │
│   └── config/
│       └── .env              # Paths, model name, constants (HF_TOKEN, DATASET_PATH, MODEL_NAME)
│
├── outputs/
│   ├── models/          # Saved BERT/saved tokenizer, sklearn .pkl, etc.
│   ├── logs/            # Training logs
│   ├── reports/         # Generated charts, confusion matrix images
│   └── inference/       # Example API outputs / predictions
│
├── requirements.txt
├── README.md
├── .gitignore
└── Dockerfile            # If deploying API
