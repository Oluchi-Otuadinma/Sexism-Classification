# Sexism Classification API

A production-ready machine learning API for detecting sexist content in text, built with FastAPI and HuggingFace Transformers.

## 🎯 Project Overview

This project provides:
- **Data Processing Pipeline**: Load, clean, and preprocess text data
- **Model Training**: Train classification models (LogReg, SVM, CNN, BERT)
- **REST API**: FastAPI service with HuggingFace inference
- **Production Features**: Caching, retry logic, batch processing, monitoring

## 📁 Project Structure

```
project-name/
│
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned and split data
│   └── external/      # External datasets
│
├── notebooks/
│   ├── 01_exploration.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning
│   ├── 03_modeling.ipynb         # Model training
│   ├── 04_evaluation.ipynb       # Model evaluation
│   └── 05_export_model.ipynb     # Export to HuggingFace
│
├── src/
│   ├── data/
│   │   ├── load_data.py          # Data loading utilities
│   │   └── preprocess.py         # Text preprocessing
│   │
│   ├── models/
│   │   ├── train.py              # Training utilities
│   │   ├── predict.py            # Prediction wrapper
│   │   └── utils.py              # Helper functions
│   │
│   ├── api/
│   │   ├── fastapi_main.py       # FastAPI application
│   │   └── hf_client.py          # HuggingFace client
│   │
│   └── config/
│       ├── settings.py           # Configuration
│       └── .env                  # Environment variables (not in git)
│
├── outputs/
│   ├── models/        # Saved models
│   ├── logs/          # Training logs
│   ├── reports/       # Evaluation reports
│   └── inference/     # API outputs
│
├── requirements.txt   # Python dependencies
├── .gitignore
├── README.md
└── Dockerfile         # For deployment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) GPU for training BERT models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sexism-classification.git
cd sexism-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `HF_API_KEY`: Your HuggingFace API key
- `HF_MODEL`: Your model name on HuggingFace
- `DATASET_PATH`: Path to your dataset

### Quick Start

1. **Prepare your data**
```python
from src.data.load_data import DataLoader
from src.data.preprocess import clean_dataframe

# Load raw data
loader = DataLoader()
df = loader.load_raw_dataset("your_dataset.csv")

# Clean text
df_clean = clean_dataframe(df, "text_column", preset="standard")

# Create train/test splits
train_df, test_df = loader.create_train_test_split(
    df_clean,
    test_size=0.2,
    stratify_column="label",
    save=True
)
```

2. **Train a model** (see notebooks/)

3. **Start the API**
```bash
uvicorn src.api.fastapi_main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test the API**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message"}'
```

## 📊 API Endpoints

### Prediction Endpoints

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Your text here"
}
```

Response:
```json
{
  "label": "sexist",
  "confidence": 0.95,
  "cached": false,
  "processing_time_ms": 245.3
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

Response:
```json
{
  "predictions": [...],
  "total_count": 3,
  "success_count": 3,
  "error_count": 0,
  "total_processing_time_ms": 450.2
}
```

#### With Confidence Threshold
```http
POST /predict?min_confidence=0.85
```

### Admin Endpoints

#### Health Check
```http
GET /health
```

#### Cache Statistics
```http
GET /cache/stats
```

#### Clear Cache
```http
POST /cache/clear
```

### API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
python test_api.py
```

## 📈 Model Training

Follow the notebooks in order:

1. **01_exploration.ipynb**: Understand your data
   - Dataset statistics
   - Label distribution
   - Text length analysis
   - Word frequency

2. **02_preprocessing.ipynb**: Clean your data
   - Remove URLs, mentions, hashtags
   - Normalize text
   - Handle imbalanced classes
   - Create train/val/test splits

3. **03_modeling.ipynb**: Train models
   - Baseline: Logistic Regression with TF-IDF
   - SVM with different kernels
   - Deep learning: CNN, BERT
   - Hyperparameter tuning

4. **04_evaluation.ipynb**: Evaluate performance
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - ROC curve
   - Error analysis

5. **05_export_model.ipynb**: Export to HuggingFace
   - Convert model to HF format
   - Push to HuggingFace Hub
   - Test inference API

## 🔧 Configuration

Configuration is managed through `src/config/settings.py` and environment variables.

Key settings:
- `PREPROCESSING_PRESET`: "minimal", "standard", or "aggressive"
- `BERT_MODEL_NAME`: Base model for fine-tuning
- `MAX_LENGTH`: Maximum sequence length
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `CACHE_SIZE`: Number of predictions to cache

See `.env.example` for all available options.

## 🎁 Features

### API Features
- ✅ Async endpoints for high concurrency
- ✅ Request/response caching (LRU)
- ✅ Retry logic with exponential backoff
- ✅ Connection pooling
- ✅ Batch prediction
- ✅ Input validation
- ✅ Comprehensive error handling
- ✅ Request logging
- ✅ Health checks
- ✅ CORS support

### Data Processing
- ✅ Multiple preprocessing presets
- ✅ Stratified train/test splits
- ✅ Class balancing options
- ✅ Text cleaning utilities
- ✅ Dataset validation

### Model Training
- ✅ Multiple model architectures
- ✅ Hyperparameter tuning
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ TensorBoard logging

## 📊 Performance

### API Performance
- Cached requests: ~50ms
- Uncached requests: ~400-700ms
- Batch (10 items): ~2000ms
- Cache hit rate: ~30-50%

### Model Performance
(Update with your model's actual performance)
- Accuracy: XX%
- F1 Score: XX%
- Precision: XX%
- Recall: XX%

## 🚢 Deployment

### Docker Deployment

1. **Build image**
```bash
docker build -t sexism-classifier .
```

2. **Run container**
```bash
docker run -p 8000:8000 \
  -e HF_API_KEY=your_key \
  -e HF_MODEL=your_model \
  sexism-classifier
```

### Production Considerations

- **Scaling**: Use multiple workers with Gunicorn
- **Rate Limiting**: Implement API rate limits
- **Monitoring**: Add Prometheus metrics
- **Logging**: Centralized logging (ELK, CloudWatch)
- **Security**: API authentication, HTTPS
- **Caching**: Redis for distributed caching

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for transformers library
- FastAPI framework
- Contributors and maintainers

## 📧 Contact

Email: ootuadinma@outlook.com

Project Link: [https://github.com/Oluchi-Otuadinma/sexism-classification](https://github.com/Oluchi-Otuadinma/sexism-classification)

## 🔗 Related Resources

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)

---
