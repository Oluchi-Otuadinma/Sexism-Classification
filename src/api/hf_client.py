"""
Enhanced Hugging Face API client with:
- Connection pooling
- Retry logic with exponential backoff
- Request timeouts
- Response caching
- Better error handling
- Structured logging
"""

import logging
import hashlib
import time
from typing import Dict, Optional
from functools import lru_cache

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from your config (adjust path as needed)
try:
    from src.config.settings import HF_API_KEY, HF_MODEL
except ImportError:
    # Fallback for testing
    import os
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    HF_MODEL = os.getenv("HF_MODEL", "")

# Constants
API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
MAX_TEXT_LENGTH = 5000  # Truncate very long texts
CACHE_SIZE = 1000  # LRU cache size
REQUEST_TIMEOUT = (10, 30)  # (connect timeout, read timeout) in seconds


class HuggingFaceClient:
    """Enhanced Hugging Face API client with connection pooling and retry logic."""
    
    def __init__(self):
        """Initialize the client with a session that has retry logic."""
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["POST"]  # Retry POST requests
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool size
            pool_maxsize=20
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        logger.info("HuggingFace client initialized with retry logic and connection pooling")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text length {len(text)} exceeds max {MAX_TEXT_LENGTH}, truncating")
            text = text[:MAX_TEXT_LENGTH]
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the input text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _cached_predict(self, text_hash: str, text: str) -> Dict:
        """
        Internal method that performs the actual API call.
        Uses text_hash as cache key for LRU cache.
        """
        logger.info(f"Cache miss for hash {text_hash[:8]}..., calling HF API")
        return self._call_api(text)
    
    def _call_api(self, text: str) -> Dict:
        """Make the actual API call to Hugging Face."""
        payload = {"inputs": text}
        
        try:
            start_time = time.time()
            response = self.session.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            elapsed_time = time.time() - start_time
            
            # Log request details
            logger.info(
                f"HF API request completed in {elapsed_time:.2f}s "
                f"(status: {response.status_code})"
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                error_msg = f"HF API error (status {response.status_code}): {response.text}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status_code": response.status_code
                }
            
            # Parse response
            result_json = response.json()
            
            # Handle different response formats
            if isinstance(result_json, list) and len(result_json) > 0:
                result = result_json[0]
            elif isinstance(result_json, dict):
                result = result_json
            else:
                error_msg = f"Unexpected response format: {type(result_json)}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Handle model loading (503 with estimated_time)
            if "error" in result and "estimated_time" in result:
                logger.warning(f"Model is loading, estimated time: {result['estimated_time']}s")
                return {
                    "error": "Model is currently loading",
                    "estimated_time": result["estimated_time"],
                    "retry_after": result["estimated_time"]
                }
            
            # Extract label and score
            label = result.get("label", "unknown")
            score = result.get("score", 0.0)
            
            return {
                "label": label,
                "confidence": score,
                "cached": False
            }
            
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timed out: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "error_type": "timeout"}
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "error_type": "connection"}
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "error_type": "request"}
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.exception(error_msg)
            return {"error": error_msg, "error_type": "unknown"}
    
    def predict(self, text: str) -> Dict:
        """
        Predict sexism classification for the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results or error information
        """
        # Validate input
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
        
        # Preprocess
        text = self._preprocess_text(text)
        
        if not text:
            return {"error": "Invalid input: text is empty after preprocessing"}
        
        # Generate cache key
        cache_key = self._get_cache_key(text)
        
        # Try to get from cache
        try:
            result = self._cached_predict(cache_key, text)
            
            # Mark as cached if this is a cache hit
            if "error" not in result:
                cache_info = self._cached_predict.cache_info()
                if cache_info.hits > 0:
                    result["cached"] = True
                    logger.info(f"Cache hit for hash {cache_key[:8]}... (hits: {cache_info.hits})")
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.exception(error_msg)
            return {"error": error_msg}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        cache_info = self._cached_predict.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) 
                        if (cache_info.hits + cache_info.misses) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._cached_predict.cache_clear()
        logger.info("Prediction cache cleared")


# Singleton instance
_client = None


def get_client() -> HuggingFaceClient:
    """Get or create the singleton HuggingFace client."""
    global _client
    if _client is None:
        _client = HuggingFaceClient()
    return _client


def hf_predict(text: str) -> Dict:
    """
    Legacy function for backward compatibility.
    Predict sexism classification using the HuggingFace API.
    
    Args:
        text: Input text to classify
        
    Returns:
        Dictionary with label and confidence, or error information
    """
    client = get_client()
    return client.predict(text)


# Example usage
if __name__ == "__main__":
    # Test the client
    test_texts = [
        "This is a test message",
        "Women should stay in the kitchen",
        "Everyone deserves equal rights"
    ]
    
    client = get_client()
    
    for text in test_texts:
        print(f"\nTesting: {text[:50]}...")
        result = client.predict(text)
        print(f"Result: {result}")
    
    # Show cache stats
    print("\nCache statistics:")
    print(client.get_cache_stats())
