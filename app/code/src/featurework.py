"""Competition-facing feature engineering wrapper.

The implementation lives in code/src/utils.py so development and submission
paths share one source of truth.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'code' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import *  # noqa: F401,F403
