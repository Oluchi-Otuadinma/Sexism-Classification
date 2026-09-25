"""Tests for the domain-aware cleaning ported from the full-pipeline notebook."""
from src.data.clean_domain_aware import clean_text_domain_aware, extract_domains


def test_extracts_domains():
    assert extract_domains("see https://www.cnn.com/x and https://bbc.co.uk/y") == ["cnn.com", "bbc.co.uk"]


def test_no_urls():
    assert extract_domains("no links here") == []


def test_clean_returns_tuple():
    text, domains = clean_text_domain_aware("Visit https://example.com/page NOW!")
    assert "example" not in text
    assert "http" not in text
    assert domains == ["example.com"]
    assert text == text.lower()


def test_stops_and_lemmatizes():
    text, _ = clean_text_domain_aware("The cats were running quickly")
    assert "the" not in text.split()
    assert "cat" in text  # lemmatized
    assert "running" in text or "run" in text
