"""Test configuration — runs before tests import the app.

1. Ensures the repository root is importable when pytest is invoked as `pytest`
   (rather than `python -m pytest`, which adds the CWD to sys.path automatically).
   CI runs plain `pytest tests/ -v`, so without this the tests fail with
   "ModuleNotFoundError: No module named 'src'".

2. Keeps tests hermetic: the API must NOT download real transformer weights
   during tests, so the model is not preloaded at startup (the lifespan
   handler skips the download; prediction is mocked per-test).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PRELOAD_MODEL", "false")
os.environ.setdefault("INFERENCE_BACKEND", "local")
os.environ.setdefault("HF_API_KEY", "test-key-not-real")
os.environ.setdefault("HF_MODEL", "test/model")
