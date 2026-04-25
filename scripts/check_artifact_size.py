"""Check local model directory and Docker image size against a GB budget."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def docker_image_size(image: str) -> int | None:
    try:
        output = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(output)


def fmt_gb(size: int) -> str:
    return f"{size / (1024 ** 3):.3f} GB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="app/model")
    parser.add_argument("--image", default="bdc2026:latest")
    parser.add_argument("--limit-gb", type=float, default=10.0)
    args = parser.parse_args()

    limit = int(args.limit_gb * (1024 ** 3))
    model_size = path_size(Path(args.model_dir))
    image_size = docker_image_size(args.image)

    print(f"model_dir={args.model_dir} size={fmt_gb(model_size)} limit={args.limit_gb:.2f} GB")
    if image_size is None:
        print(f"image={args.image} size=unavailable")
    else:
        print(f"image={args.image} size={fmt_gb(image_size)} limit={args.limit_gb:.2f} GB")

    failed = model_size > limit or (image_size is not None and image_size > limit)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
