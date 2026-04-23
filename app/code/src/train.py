"""Competition-facing training entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'code' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import main


if __name__ == '__main__':
    main()
