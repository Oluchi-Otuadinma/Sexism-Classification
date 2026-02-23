"""
Text preprocessing utilities for sexism classification.

Includes functions for cleaning, normalizing, and tokenizing text data.
"""

import re
import logging
from typing import List, Optional, Callable

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing pipeline for NLP tasks.
    
    Provides methods for cleaning, normalizing, and tokenizing text.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        min_word_length: int = 2
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            min_word_length: Minimum word length to keep
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Initialize NLTK components if needed
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_url(self, text: str) -> str:
        """Remove URLs from text."""
        # Pattern matches http://, https://, www., and common TLDs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = re.sub(r'www\.\S+', '', text)
        return text
    
    def clean_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@\w+', '', text)
    
    def clean_hashtags(self, text: str) -> str:
        """Remove #hashtags from text."""
        return re.sub(r'#\w+', '', text)
    
    def clean_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def clean_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return re.sub(r'[^\w\s]', ' ', text)
    
    def clean_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        return ' '.join(text.split())
    
    def process_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Apply cleaning steps
        if self.lowercase:
            text = text.lower()
        
        if self.remove_urls:
            text = self.clean_url(text)
        
        if self.remove_mentions:
            text = self.clean_mentions(text)
        
        if self.remove_hashtags:
            text = self.clean_hashtags(text)
        
        if self.remove_numbers:
            text = self.clean_numbers(text)
        
        if self.remove_punctuation:
            text = self.clean_punctuation(text)
        
        # Tokenize if needed for advanced processing
        if self.remove_stopwords or self.lemmatize or self.min_word_length > 1:
            try:
                tokens = word_tokenize(text)
                
                # Filter stopwords
                if self.remove_stopwords:
                    tokens = [t for t in tokens if t.lower() not in self.stop_words]
                
                # Apply lemmatization
                if self.lemmatize:
                    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                
                # Filter by word length
                if self.min_word_length > 1:
                    tokens = [t for t in tokens if len(t) >= self.min_word_length]
                
                text = ' '.join(tokens)
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}")
        
        # Final cleanup
        text = self.clean_extra_whitespace(text)
        
        return text.strip()
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: Optional[str] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Apply preprocessing to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name for output column (default: text_column + '_clean')
            inplace: Whether to modify DataFrame in place
            
        Returns:
            DataFrame with processed text column
        """
        if not inplace:
            df = df.copy()
        
        if output_column is None:
            output_column = f"{text_column}_clean"
        
        logger.info(f"Processing {len(df):,} texts...")
        
        # Apply preprocessing
        df[output_column] = df[text_column].apply(self.process_text)
        
        # Log statistics
        original_lengths = df[text_column].str.len()
        cleaned_lengths = df[output_column].str.len()
        
        logger.info(
            f"Average length - Original: {original_lengths.mean():.1f}, "
            f"Cleaned: {cleaned_lengths.mean():.1f}"
        )
        
        # Check for empty results
        n_empty = (df[output_column].str.strip() == '').sum()
        if n_empty > 0:
            logger.warning(f"{n_empty} texts became empty after preprocessing")
        
        return df


# Preset configurations
def get_minimal_preprocessor() -> TextPreprocessor:
    """Get preprocessor with minimal cleaning (lowercase, URLs, extra spaces)."""
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=False,
        remove_hashtags=False,
        remove_numbers=False,
        remove_punctuation=False,
        remove_stopwords=False,
        lemmatize=False
    )


def get_standard_preprocessor() -> TextPreprocessor:
    """Get preprocessor with standard cleaning for classification."""
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_numbers=True,
        remove_punctuation=False,
        remove_stopwords=False,
        lemmatize=False
    )


def get_aggressive_preprocessor() -> TextPreprocessor:
    """Get preprocessor with aggressive cleaning (all options enabled)."""
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        remove_numbers=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        min_word_length=3
    )


# Convenience functions
def clean_text(
    text: str,
    preset: str = "standard"
) -> str:
    """
    Clean a single text string.
    
    Args:
        text: Input text
        preset: Preprocessing preset ('minimal', 'standard', or 'aggressive')
        
    Returns:
        Cleaned text
    """
    if preset == "minimal":
        preprocessor = get_minimal_preprocessor()
    elif preset == "aggressive":
        preprocessor = get_aggressive_preprocessor()
    else:
        preprocessor = get_standard_preprocessor()
    
    return preprocessor.process_text(text)


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
    preset: str = "standard",
    **kwargs
) -> pd.DataFrame:
    """
    Clean text in a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text
        preset: Preprocessing preset ('minimal', 'standard', or 'aggressive')
        **kwargs: Additional arguments for process_dataframe
        
    Returns:
        DataFrame with cleaned text
    """
    if preset == "minimal":
        preprocessor = get_minimal_preprocessor()
    elif preset == "aggressive":
        preprocessor = get_aggressive_preprocessor()
    else:
        preprocessor = get_standard_preprocessor()
    
    return preprocessor.process_dataframe(df, text_column, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test preprocessing
    sample_texts = [
        "Check this out! https://example.com #awesome @user",
        "Women should stay in the kitchen... disgusting!!!",
        "Everyone deserves equal rights and opportunities 123",
        "RT @someone: This is a retweet with numbers 12345"
    ]
    
    print("Testing different preprocessing levels:\n")
    
    for preset in ["minimal", "standard", "aggressive"]:
        print(f"\n{preset.upper()} PREPROCESSING:")
        print("-" * 50)
        
        for text in sample_texts:
            cleaned = clean_text(text, preset=preset)
            print(f"Original: {text}")
            print(f"Cleaned:  {cleaned}\n")
    
    # Test on DataFrame
    print("\nTesting DataFrame processing:")
    print("-" * 50)
    
    df = pd.DataFrame({
        "text": sample_texts,
        "label": [0, 1, 0, 1]
    })
    
    df_cleaned = clean_dataframe(df, "text", preset="standard")
    print(df_cleaned[["text", "text_clean"]].head())