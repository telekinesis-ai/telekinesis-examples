"""
Minimal detection dataset logger example.

Generates synthetic RGB frames with hand-placed bounding-box annotations and
writes them to disk using ``DetectionLogger``.

Supported output formats:

- YOLO
- RF-DETR / COCO

No server or external data is required.

Usage:
    python logger_inline.py
"""

from pathlib import Path

import numpy as np
from loguru import logger

from telekinesis.dataengine.data_loggers import DetectionLogger

OUTPUT_PATH = Path("results/inline_dataset")
OUTPUT_FORMAT = "yolo"
MODE = "overwrite"
NUM_FRAMES = 20

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

CATEGORIES = [
    {
        "id": 1,
        "name": "class_a",
        "supercategory": "object",
    },
    {
        "id": 2,
        "name": "class_b",
        "supercategory": "object",
    },
]


def make_sample(
    frame_index: int,
) -> tuple[np.ndarray, list[dict]]:
    """Create one synthetic image and its annotations.

    Args:
        frame_index: Index used to seed the random number generator.

    Returns:
        RGB image and COCO-style annotations.
    """
    random = np.random.default_rng(frame_index)

    image = random.integers(
        0,
        255,
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )

    annotations = [
        {
            "category_id": 1,
            "bbox": [
                float(random.integers(0, IMAGE_WIDTH // 2)),
                float(random.integers(0, IMAGE_HEIGHT // 2)),
                120.0,
                90.0,
            ],
        },
        {
            "category_id": 2,
            "bbox": [
                float(
                    random.integers(
                        IMAGE_WIDTH // 2,
                        IMAGE_WIDTH - 80,
                    )
                ),
                float(
                    random.integers(
                        IMAGE_HEIGHT // 2,
                        IMAGE_HEIGHT - 60,
                    )
                ),
                60.0,
                50.0,
            ],
        },
    ]

    return image, annotations


def logger_inline_example() -> None:
    """Log synthetic detection samples into a dataset."""
    dataset_logger = DetectionLogger.create(
        OUTPUT_FORMAT,
        OUTPUT_PATH,
        CATEGORIES,
        mode=MODE,
    )

    for frame_index in range(NUM_FRAMES):
        image, annotations = make_sample(frame_index)

        dataset_logger.log(
            image,
            annotations,
        )

    dataset_logger.close()

    logger.success(
        f"{OUTPUT_FORMAT} dataset written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    logger_inline_example()