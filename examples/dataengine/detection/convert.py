"""
Convert a detection dataset between the YOLO and COCO / RF-DETR layouts.

Images are copied byte-for-byte; only the annotation geometry is rewritten, so
the conversion is lossless and fast. Boxes and segmentation polygons are both
supported (pick with --task). Splits and class names are preserved.

Usage:
    # COCO / RF-DETR  ->  YOLO, into results/converted_dataset
    python convert.py --input-path results/detection_rfdetr --to yolo

    # YOLO  ->  COCO / RF-DETR, to a specific destination
    python convert.py --input-path results/detection_yolo \\
        --output-path results/yolo_to_coco --to coco

    # Segmentation polygons, overwriting an existing destination
    python convert.py --input-path results/seg_yolo \\
        --output-path results/seg_coco --to coco --task segment --overwrite

Layouts:
    YOLO (Ultralytics)              COCO / RF-DETR
        <root>/                         <root>/
            images/<split>/                 <split>/  (images + json)
            labels/<split>/                     _annotations.coco.json
            data.yaml

The source format is auto-detected; pass --input-format to override it.
"""

import argparse
import pathlib

from telekinesis.dataengine.detection.utils import convert_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a detection dataset between YOLO and COCO/RF-DETR."
    )
    parser.add_argument(
        "--input-path",
        type=pathlib.Path,
        required=True,
        help="Source dataset directory",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default="results/converted_dataset",
        help="Output directory for the converted dataset "
        "(default: results/converted_dataset)",
    )
    parser.add_argument(
        "--to",
        dest="to_format",
        choices=("coco", "rfdetr", "yolo"),
        required=True,
        help="Target format ('coco'/'rfdetr' are synonyms)",
    )
    parser.add_argument(
        "--input-format",
        choices=("coco", "rfdetr", "yolo"),
        default=None,
        help="Source format (auto-detected if omitted)",
    )
    parser.add_argument(
        "--task",
        choices=("detect", "segment"),
        default="detect",
        help="Convert bounding boxes ('detect') or polygons ('segment')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination if it exists and is non-empty",
    )
    args = parser.parse_args()

    convert_dataset(
        args.input_path,
        args.output_path,
        args.to_format,
        src_format=args.input_format,
        task=args.task,
        overwrite=args.overwrite,
    )
    print(f"Converted dataset written to {args.output_path}")
