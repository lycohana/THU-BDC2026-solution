"""Competition-facing prediction entrypoint."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'code' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import config
from predict import main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--postprocess', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.postprocess:
        config.setdefault('postprocess', {})['filter'] = args.postprocess
    main()
