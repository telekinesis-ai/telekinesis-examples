"""
Merge several detection datasets into a single dataset.

Each input may be YOLO or COCO / RF-DETR — the format is auto-detected per
input, so mixed sets are fine. Category ids in the merged dataset are unified by
class *name*: the same class name gets the same id in every input, distinct
names get distinct 1-based ids (assigned first-seen, following the order of the
inputs). Images are copied byte-for-byte with a per-dataset prefix so identical
file names don't collide, and image / annotation ids are reassigned to be
globally unique.

Usage:
    python merge.py \\
        --input-path results/dataset_a results/dataset_b results/dataset_c
"""

from __future__ import annotations

import argparse
import pathlib

from telekinesis.dataengine.detection.utils import merge_datasets


def main(
    input_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    overwrite: bool = False,
) -> None:
    summary = merge_datasets(input_paths, output_path, overwrite=overwrite)
    print(f"Merged dataset written to {output_path}")
    print(f"  classes: {summary['num_classes']}")
    print(f"  splits : {summary['counts']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge several detection datasets into one."
    )
    parser.add_argument(
        "--input-path",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Two or more source dataset directories to merge",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default="results/merged_dataset",
        help="Output directory for the merged dataset "
        "(default: results/merged_dataset)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination if it exists and is non-empty",
    )
    args = parser.parse_args()

    main(
        input_paths=args.input_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )
