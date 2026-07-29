"""
Visualize a YOLO or RF-DETR dataset produced by DetectionLogger.

Loads the dataset into FiftyOne and launches the interactive app. The format is
auto-detected from the directory layout.

Usage:
    python visualize.py --input-path results/detection_yolo
"""

from __future__ import annotations

import argparse
import pathlib

from telekinesis.dataengine.detection.utils import visualize


def main(
    input_path: pathlib.Path,
    max_samples: int | None = None,
) -> None:
    visualize(input_path, max_samples=max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a detection dataset.")
    parser.add_argument(
        "--input-path",
        type=pathlib.Path,
        required=True,
        help="Dataset directory to visualize",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to load (default: 1000)",
    )
    args = parser.parse_args()

    main(input_path=args.input_path, max_samples=args.max_samples)
