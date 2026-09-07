"""
Visualize a detection dataset.

Loads a YOLO or RF-DETR dataset produced by ``DetectionLogger`` into
FiftyOne and launches the interactive app.

Update ``INPUT_PATH`` below to point to the dataset you want to visualize.

Usage:
    python visualize_dataset.py
"""

from pathlib import Path

from telekinesis.dataengine.data_loggers import utils


INPUT_PATH = Path("results/pipeline/yolo_dataset")
MAX_SAMPLES = 1000


def visualize_dataset_example() -> None:
    """Visualize the configured detection dataset in FiftyOne."""
    utils.visualize(
        INPUT_PATH,
        max_samples=MAX_SAMPLES,
    )


if __name__ == "__main__":
    visualize_dataset_example()