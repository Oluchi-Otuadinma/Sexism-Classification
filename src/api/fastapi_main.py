"""
Enhanced FastAPI application for sexism detection with:
- Async endpoints
- Input validation
- Batch prediction
- CORS support
- Request logging
- Health checks
- OpenAPI documentation
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.api.hf_client import get_client
from src.api.model_manager import ModelManager
from src.config.settings import (
    CACHE_SIZE,
    INFERENCE_BACKEND,
    LOCAL_MODEL_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_ID,
    PRELOAD_MODEL,
)

# Configure logging (level/format from settings — single source of truth)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup and release it at shutdown.

    Local backend (default): creates a ModelManager which downloads the
    pre-trained Transformer weights (vinai/bertweet-base) and attaches a
    fresh classification head (a linear layer outputting probabilities
    across the target classes), unless a fine-tuned checkpoint exists at
    LOCAL_MODEL_DIR. Set PRELOAD_MODEL=false to defer the download until
    the first request.

    Set INFERENCE_BACKEND=hf_api to use the HuggingFace Inference API
    instead of running the model locally.
    """
    if INFERENCE_BACKEND == "hf_api":
        logger.info("Inference backend: HuggingFace Inference API")
        app.state.model_manager = get_client()
    else:
        logger.info("Inference backend: local transformer")
        manager = ModelManager(
            model_id=MODEL_ID,
            num_labels=2,
            checkpoint_dir=LOCAL_MODEL_DIR,
            cache_size=CACHE_SIZE,
        )
        if PRELOAD_MODEL:
            manager.load()
        else:
            logger.info("PRELOAD_MODEL=false — model will load on first request")
        app.state.model_manager = manager

    yield

    # Shutdown: release model memory
    manager = getattr(app.state, "model_manager", None)
    if manager is not None and hasattr(manager, "shutdown"):
        manager.shutdown()


# Initialize FastAPI app with metadata
app = FastAPI(
    title="Sexism Detection API",
    description="API for detecting sexist content in text using machine learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class TextInput(BaseModel):
    """Single text input for prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sexist content",
        json_schema_extra={
            "example": "This is a sample text for classification"
        }
    )
    
    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v):
        """Validate that text is not just whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class BatchTextInput(BaseModel):
    """Multiple texts for batch prediction."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze",
        json_schema_extra={
            "example": ["Text 1", "Text 2", "Text 3"]
        }
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """Validate each text in the batch."""
        validated = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f'Item {i} must be a string')
            text = text.strip()
            if not text:
                raise ValueError(f'Item {i} cannot be empty')
            if len(text) > 5000:
                raise ValueError(f'Item {i} exceeds maximum length of 5000 characters')
            validated.append(text)
        return validated


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    label: str = Field(..., description="Predicted label (e.g., 'sexist' or 'not sexist')")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    cached: bool = Field(False, description="Whether result was served from cache")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    total_count: int
    success_count: int
    error_count: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    cache_stats: dict
    model: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_type: Optional[str] = None
    details: Optional[dict] = None


# Global variables for tracking
start_time = time.time()


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start = time.time()
    
    # Log request
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"(status: {response.status_code}, duration: {duration:.3f}s)"
    )
    
    return response


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sexism Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health(request: Request):
    """
    Health check endpoint.
    
    Returns API status, version, uptime, model info, and cache statistics.
    """
    try:
        manager = request.app.state.model_manager
        cache_stats = manager.get_cache_stats()
        model_info = manager.info() if hasattr(manager, "info") else {"backend": "hf_api"}
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime_seconds": time.time() - start_time,
            "cache_stats": cache_stats,
            "model": model_info,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model unavailable"}
    },
    tags=["Prediction"]
)
async def predict(
    input: TextInput,
    request: Request,
    min_confidence: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0-1). Results below this are filtered out."
    )
):
    """
    Predict whether a single text contains sexist content.
    
    - **text**: The text to analyze (required, 1-5000 characters)
    - **min_confidence**: Optional confidence threshold for filtering results
    
    Returns the predicted label, confidence score, and whether the result was cached.
    """
    try:
        start = time.time()
        
        # Get prediction from the model loaded at startup (lifespan)
        manager = request.app.state.model_manager
        result = manager.predict(input.text)
        
        processing_time = (time.time() - start) * 1000  # Convert to milliseconds
        
        # Check for errors
        if "error" in result:
            status_code = result.get("status_code", 500)
            
            # Handle model loading
            if "estimated_time" in result:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": result["error"],
                        "error_type": "model_loading",
                        "details": {
                            "estimated_time": result["estimated_time"],
                            "retry_after": result.get("retry_after")
                        }
                    }
                )
            
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": result["error"],
                    "error_type": result.get("error_type", "unknown")
                }
            )
        
        # Apply confidence threshold if specified
        if min_confidence is not None and result["confidence"] < min_confidence:
            return JSONResponse(
                status_code=200,
                content={
                    "label": "uncertain",
                    "confidence": result["confidence"],
                    "cached": result.get("cached", False),
                    "processing_time_ms": processing_time,
                    "note": f"Confidence {result['confidence']:.3f} below threshold {min_confidence}"
                }
            )
        
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "cached": result.get("cached", False),
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during prediction",
                "error_type": "server_error"
            }
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"]
)
async def predict_batch(input: BatchTextInput, request: Request):
    """
    Predict sexist content for multiple texts in one request.
    
    - **texts**: List of texts to analyze (1-100 texts)
    
    Returns predictions for all texts with aggregate statistics.
    This is more efficient than making multiple single predictions.
    """
    try:
        start = time.time()
        
        manager = request.app.state.model_manager
        predictions = []
        success_count = 0
        error_count = 0
        
        # Process each text
        for text in input.texts:
            try:
                result = manager.predict(text)
                
                if "error" in result:
                    error_count += 1
                    predictions.append({
                        "label": "error",
                        "confidence": 0.0,
                        "cached": False,
                        "error": result["error"]
                    })
                else:
                    success_count += 1
                    predictions.append({
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "cached": result.get("cached", False)
                    })
            except Exception as e:
                error_count += 1
                predictions.append({
                    "label": "error",
                    "confidence": 0.0,
                    "cached": False,
                    "error": str(e)
                })
        
        total_time = (time.time() - start) * 1000
        
        return {
            "predictions": predictions,
            "total_count": len(predictions),
            "success_count": success_count,
            "error_count": error_count,
            "total_processing_time_ms": total_time
        }
        
    except Exception as e:
        logger.exception(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during batch prediction",
                "error_type": "server_error"
            }
        )


@app.post("/cache/clear", tags=["Admin"])
async def clear_cache(request: Request):
    """
    Clear the prediction cache.
    
    Use this endpoint to force fresh predictions for all texts.
    """
    try:
        request.app.state.model_manager.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.exception(f"Cache clear failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to clear cache"}
        )


@app.get("/cache/stats", tags=["Admin"])
async def cache_stats(request: Request):
    """
    Get cache statistics.
    
    Returns information about cache hits, misses, and hit rate.
    """
    try:
        return request.app.state.model_manager.get_cache_stats()
    except Exception as e:
        logger.exception(f"Cache stats failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get cache stats"}
        )


# Run with: uvicorn src.api.fastapi_main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
