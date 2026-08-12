"""
Minimal detection-dataset logger example.

Generates a few synthetic frames + annotations and writes them to disk in one of
the supported training formats (pick with --format):

  - YOLO    → images/, labels/, data.yaml
  - RF-DETR → <split>/_annotations.coco.json (COCO)

The dataset is written straight into --output-path.

No servers required — the frames here are random noise with hand-placed boxes.
Images are raw ``np.ndarray``; categories and annotations are plain COCO-style
JSON (lists/dicts).

Usage:
    # YOLO into results/inline_dataset
    python logger_inline.py

    # RF-DETR, appending to an existing dataset elsewhere
    python logger_inline.py --output-path results/detection_rfdetr \\
        --format rfdetr --mode append
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from loguru import logger

from telekinesis.dataengine import DetectionLogger


# The class table shared by every sample, in COCO ``categories`` JSON form.
# category_id -> 0-based class index is derived from the order below
# (so id 1 -> class 0, id 2 -> class 1).
CATEGORIES = [
    {"id": 1, "name": "class_a", "supercategory": "object"},
    {"id": 2, "name": "class_b", "supercategory": "object"},
]


# =============================================================================
# SYNTHETIC SAMPLE GENERATOR
# =============================================================================


def make_sample(frame_index: int) -> tuple[np.ndarray, list[dict]]:
    """Return one (image, annotations) pair of fake data.

    ``image`` is a raw ``HxWx3`` uint8 RGB array; ``annotations`` is a list of
    COCO annotation dicts (``category_id`` + ``bbox`` as ``[x, y, w, h]``).
    """
    image_height, image_width = 480, 640
    random = np.random.default_rng(frame_index)
    image = random.integers(0, 255, (image_height, image_width, 3), dtype=np.uint8)

    # Two COCO-style [x, y, w, h] boxes, one per class.
    annotations = [
        {
            "category_id": 1,
            "bbox": [
                float(random.integers(0, image_width // 2)),
                float(random.integers(0, image_height // 2)),
                120.0,
                90.0,
            ],
        },
        {
            "category_id": 2,
            "bbox": [
                float(random.integers(image_width // 2, image_width - 80)),
                float(random.integers(image_height // 2, image_height - 60)),
                60.0,
                50.0,
            ],
        },
    ]
    return image, annotations


# =============================================================================
# MAIN
# =============================================================================


def main(
    output_format: str,
    output_path: pathlib.Path,
    mode: str,
    num_frames: int = 20,
) -> None:
    dataset_logger = DetectionLogger.create(
        output_format, output_path, CATEGORIES, mode=mode
    )

    # Splits are assigned by the logger itself — 80/10/10 train/val/test by default.
    for frame_index in range(num_frames):
        image, annotations = make_sample(frame_index)
        dataset_logger.log(image, annotations)  # split auto-assigned 80/10/10

    dataset_logger.close()  # writes data.yaml (YOLO) / _annotations.coco.json (RF-DETR)
    logger.success(f"{output_format} dataset -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log a synthetic detection dataset in YOLO or RF-DETR format."
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default="results/inline_dataset",
        help="Dataset directory (default: results/inline_dataset)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("yolo", "rfdetr"),
        default="yolo",
        help="Dataset format to write (default: yolo)",
    )
    parser.add_argument(
        "--mode",
        choices=("create", "overwrite", "append"),
        default="create",
        help="How to handle an existing non-empty dataset directory "
        "(default: create)",
    )
    args = parser.parse_args()

    main(
        output_format=args.output_format,
        output_path=args.output_path,
        mode=args.mode,
    )
