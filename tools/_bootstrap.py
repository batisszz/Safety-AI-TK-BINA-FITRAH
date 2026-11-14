"""Membuat 'python tools/xxx.py' bisa jalan tanpa 'pip install -e .' (menambahkan src/ ke sys.path)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
