"""
Convert a detection dataset between YOLO and COCO / RF-DETR layouts.

Images are copied byte-for-byte; only annotation geometry is rewritten.
Bounding-box and segmentation datasets are supported.

The source format is auto-detected unless ``INPUT_FORMAT`` is set explicitly.

Run example after generating a YOLO dataset with the logger_inline.py script.

Usage:
    python convert.py
"""

from pathlib import Path

from telekinesis.dataengine.data_loggers import utils

INPUT_PATH = Path("results/pipeline/yolo_dataset")
OUTPUT_PATH = Path("results/converted_dataset")

TO_FORMAT = "rfdetr"
INPUT_FORMAT = None
TASK = "detect"
OVERWRITE = True


def convert_dataset_example() -> None:
    """Convert the configured detection dataset."""
    utils.convert_dataset(
        INPUT_PATH,
        OUTPUT_PATH,
        TO_FORMAT,
        src_format=INPUT_FORMAT,
        task=TASK,
        overwrite=OVERWRITE,
    )

    print(f"Converted dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_dataset_example()