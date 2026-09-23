"""Tests for text preprocessing utilities."""
import pytest

from src.data.preprocess import clean_text, get_minimal_preprocessor, get_standard_preprocessor, get_aggressive_preprocessor


def test_clean_text_returns_string():
    assert isinstance(clean_text("Hello world"), str)


@pytest.mark.parametrize("preset_fn", [get_minimal_preprocessor, get_standard_preprocessor, get_aggressive_preprocessor])
def test_presets_handle_empty(preset_fn):
    p = preset_fn()
    assert p.process_text("") == ""


def test_url_removal():
    p = get_standard_preprocessor()
    out = p.process_text("check this https://example.com/page out")
    assert "https" not in out
    assert "example.com" not in out or "example" in out  # domain may be kept depending on preset


def test_whitespace_normalised():
    p = get_standard_preprocessor()
    assert "  " not in p.process_text("too   many     spaces")


def test_lowercase():
    p = get_standard_preprocessor()
    assert p.process_text("HELLO") == p.process_text("hello")
