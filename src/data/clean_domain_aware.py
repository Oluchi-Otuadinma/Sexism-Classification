"""
Domain-aware text cleaning — ported from notebooks/06_full_pipeline.ipynb
(X3a7e2118_1_066.ipynb, the full coursework pipeline).

Extends src.data.preprocess with URL-domain feature extraction: URLs are
stripped from the text and their registrable domains (e.g. "cnn.com") are
returned separately, so they can be used as auxiliary features downstream
(MultiLabelBinarizer -> hstack with TF-IDF).
"""

import re
from typing import List, Tuple

import nltk
import tldextract
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _res in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords", "corpora/wordnet"):
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/", 1)[1].split("/")[0] if "tokenizers" in _res else _res.split("/", 1)[1], quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

_URL_PATTERN = r"https?://\S+|www\.\S+"


def extract_domains(text: str) -> List[str]:
    """Return the registrable domains of all URLs in *text* (without modifying the text)."""
    return [f"{e.domain}.{e.suffix}" for e in (tldextract.extract(m) for m in re.findall(_URL_PATTERN, text)) if e.suffix]


def clean_text_domain_aware(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> Tuple[str, List[str]]:
    """
    Clean *text* and extract URL domains in one pass.

    Returns:
        (cleaned_text, domains) — the cleaned text with URLs removed, and the
        list of registrable domains that appeared in it (may be empty).
    """
    domains: List[str] = []
    for match in re.findall(_URL_PATTERN, text):
        extracted = tldextract.extract(match)
        domain = f"{extracted.domain}.{extracted.suffix}"  # e.g. "cnn.com"
        if extracted.suffix:
            domains.append(domain)
        text = text.replace(match, "")

    if lowercase:
        text = text.lower()

    # Remove special characters, punctuation, and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [w for w in tokens if w not in _stop_words]
    if lemmatize:
        tokens = [(_lemmatizer.lemmatize(w)) for w in tokens]

    return " ".join(tokens), domains


def clean_dataframe_domain_aware(df, text_column: str = "text"):
    """
    Apply :func:`clean_text_domain_aware` over a DataFrame.

    Replaces ``df[text_column]`` with the cleaned text and adds a
    ``domains`` column (list of URL domains per row).
    """
    import pandas as pd

    results = df[text_column].apply(clean_text_domain_aware)
    df = df.copy()
    df[[text_column, "domains"]] = pd.DataFrame(results.tolist(), index=df.index)
    return df
