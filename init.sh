#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/model /app/output /app/temp model output temp

python - <<'PY'
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
try:
    import torch
    print("Torch:", torch.__version__)
except Exception as exc:
    print("Torch unavailable:", exc)
PY
