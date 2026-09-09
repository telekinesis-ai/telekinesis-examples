"""Example to show how to merge several detection datasets into a single dataset.

Run after generating the individual detection datasets.
Run below to generate datasets
python detection_pipeline   .py

For eg: 
    python merge_datasets.py
"""

from pathlib import Path

from telekinesis.dataengine.data_loggers.detection.utils import merge_datasets


INPUT_PATHS = [
    Path("results/pipeline/yolo_converted"),
    Path("results/pipeline/rfdetr_dataset"),
]

OUTPUT_PATH = Path("results/merged_dataset")
OVERWRITE = True


def merge_datasets_example() -> None:
    """Merge the configured detection datasets into one dataset."""
    summary = merge_datasets(
        INPUT_PATHS,
        OUTPUT_PATH,
        overwrite=OVERWRITE,
    )

    print(f"Merged dataset written to {OUTPUT_PATH}")
    print(f"  classes: {summary['num_classes']}")
    print(f"  splits : {summary['counts']}")


if __name__ == "__main__":
    merge_datasets_example()