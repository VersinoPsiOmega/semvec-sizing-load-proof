"""Test setup: makes the three deliverable modules importable (not packages)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("sizing", "harness", "load"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
