"""Root conftest.py — adds PrecipModels/ to sys.path so tests can import top-level modules."""
import sys
from pathlib import Path

# Ensure PrecipModels/ (this directory) is importable as a package root.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
