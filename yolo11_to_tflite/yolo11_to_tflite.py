"""
Export a YOLO11 PyTorch checkpoint to TFLite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11 .pt model to TFLite")
    parser.add_argument(
        "input",
        help="Input YOLO .pt file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.input).expanduser().resolve()

    model = YOLO(str(model_path))

    model.export(format="tflite", dynamic=False)


if __name__ == "__main__":
    main()
