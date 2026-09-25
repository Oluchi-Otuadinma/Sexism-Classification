"""Ensure the repository root is importable when pytest is invoked as `pytest`
(rather than `python -m pytest`, which adds the CWD to sys.path automatically).
CI runs plain `pytest tests/ -v`, so without this the tests fail with
"ModuleNotFoundError: No module named 'src'".
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
