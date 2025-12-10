import re
import tldextract
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ----------------------------------------------------------------------
# Ensure NLTK packages exist (safe to run even if already downloaded)
# ----------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialise NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text and extract domains
def clean_text(
    text: str,
    lowercase: bool =True, 
    replace_urls: bool = True, 
    extract_domain: bool = False, 
    remove_stopwords: bool = True, 
    lemmatize: bool = True,
):
    """Clean and normalize text while optionally extracting URL domains.

    Returns:
        cleaned_text (str)
        domains (list[str]) — only if extract_domain=True
    """

    if not isinstance(text, str):
        text = str(text)
      
    # --- 1. Lowercase -----------------------------------------------------------------------
    if lowercase:
        text = text.lower()

    # --- 2. Extract and remove URLs (OPTIONAL - Commented Out) ------------------------------
    # Extract domain names from URLs
    # url_pattern = r'https?://\S+|www\.\S+'
    # domains = []  # Store extracted domains
  
    # matches = re.findall(url_pattern, text)

    # for match in matches:
    #     extracted = tldextract.extract(match)
    #     domain = f"{extracted.domain}.{extracted.suffix}"  # e.g., "cnn.com"
    #     domains.append(domain)  # Save domain for analysis
        
    #     if replace_urls:      
    #         text = text.replace(match, "")  # Remove the URL from text 
  
    # --- 3. Remove special characters, punctuation, and numbers -----------------------------
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # --- 4. Tokenization --------------------------------------------------------------------
    tokens = word_tokenize(text)

    # --- 5. Remove stopwords ----------------------------------------------------------------
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # --- 6. Lemmatization -------------------------------------------------------------------
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # --- 7. Join back into a clean string --------------------------------
    cleaned_text = " ".join(tokens)

    # --- Output handling --------------------------------------------------
    
    # if extract_domain:
    #     return cleaned_text, domains
    # else:
    return cleaned_text

