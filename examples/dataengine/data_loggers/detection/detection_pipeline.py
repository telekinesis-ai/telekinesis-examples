"""
End-to-end detection dataset pipeline.

Demonstrates how to:

1. Detect objects in images using Grounding DINO.
2. Log detections into YOLO and RF-DETR datasets.
3. Convert the YOLO dataset to RF-DETR/COCO.
4. Merge the resulting datasets.
5. Visualize the merged dataset.

All outputs are written under ``--output-path``.

Usage:
    python tutorial.py
    python tutorial.py --prompt "forklift, pallet"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from telekinesis import datatypes, retina
from telekinesis.dataengine.data_loggers import DetectionLogger, utils


IMAGE_URLS = [
    "https://assets.telekinesis.ai/examples/v1/images/warehouse_1.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/warehouse_2.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/palletizing.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/cartons_arranged.png",
    "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg",
]


def detect_objects(
    image: datatypes.Image,
    objects: list[str],
    name_to_id: dict[str, int],
) -> tuple[np.ndarray, list[dict]]:
    """Detect objects in an image and return COCO-style annotations.

    Grounding DINO assigns category IDs independently for each request.
    The detected category names are therefore remapped to the stable IDs
    defined by ``name_to_id``.

    Args:
        image: Input image.
        objects: Object names to detect.
        name_to_id: Mapping from category names to dataset category IDs.

    Returns:
        Image as a NumPy array and its annotations.
    """
    detections, categories = retina.detect_objects_using_grounding_dino(
        image=image,
        objects=objects,
        box_threshold=0.3,
        text_threshold=0.25,
    )

    local_names = dict(
        zip(
            categories.ids.tolist(),
            categories.names.tolist(),
        )
    )

    annotations = []

    for detection in detections:
        name = local_names.get(detection.category_id)
        category_id = name_to_id.get(name)

        if category_id is None:
            logger.warning(
                f"Unexpected class {name!r} from Grounding DINO. Skipping."
            )
            continue

        annotations.append(
            {
                "bbox": detection.bbox.tolist(),
                "category_id": category_id,
            }
        )

    return image.to_numpy(), annotations

def collect_samples(
    image_urls: list[str],
    objects: list[str],
    name_to_id: dict[str, int],
) -> list[tuple[np.ndarray, list[dict]]]:
    """Detect objects in all input images."""
    samples = []

    for url in image_urls:
        image = datatypes.Image.from_url(url)

        sample = detect_objects(
            image=image,
            objects=objects,
            name_to_id=name_to_id,
        )

        samples.append(sample)

        logger.success(
            f"Detected {len(sample[1])} objects in {url}"
        )

    return samples

def log_dataset(
    output_format: str,
    output_path: Path,
    categories: list[dict],
    samples: list[tuple[np.ndarray, list[dict]]],
) -> None:
    """Log detection samples into a dataset."""
    dataset_logger = DetectionLogger.create(
        output_format,
        output_path,
        categories,
        mode="overwrite",
    )

    for image, annotations in samples:
        dataset_logger.log(
            image=image,
            annotations=annotations,
        )

    dataset_logger.close()

def create_datasets(
    samples: list[tuple[np.ndarray, list[dict]]],
    categories: list[dict],
    yolo_path: Path,
    rfdetr_path: Path,
) -> None:
    """Create separate YOLO and RF-DETR datasets."""
    split_index = len(samples) // 2

    yolo_samples = samples[:split_index]
    rfdetr_samples = samples[split_index:]

    log_dataset(
        output_format="yolo",
        output_path=yolo_path,
        categories=categories,
        samples=yolo_samples,
    )

    log_dataset(
        output_format="rfdetr",
        output_path=rfdetr_path,
        categories=categories,
        samples=rfdetr_samples,
    )

def convert_dataset(
    source_path: Path,
    output_path: Path,
) -> None:
    """Convert a dataset to RF-DETR/COCO format."""
    utils.convert_dataset(
        source_path,
        output_path,
        "rfdetr",
        overwrite=True,
    )

def merge_datasets(
    dataset_paths: list[Path],
    output_path: Path,
) -> dict:
    """Merge multiple RF-DETR/COCO datasets."""
    return utils.merge_datasets(
        dataset_paths,
        output_path,
        overwrite=True,
    )


def detection_pipeline_example(
    output_path: Path,
    prompt: str,
    image_urls: list[str],
) -> None:
    """Run the complete detection dataset pipeline."""

    # ==== Step 0: Prepare paths and categories ====
    class_names = [
        name.strip()
        for name in prompt.split(",")
        if name.strip()
    ]
    categories = [
        {
            "id": index + 1,
            "name": name,
            "supercategory": "object",
        }
        for index, name in enumerate(class_names)
    ]
    name_to_id = {
        category["name"]: category["id"]
        for category in categories
    }
    yolo_path = output_path / "yolo_dataset"
    rfdetr_path = output_path / "rfdetr_dataset"
    converted_path = output_path / "yolo_converted"
    merged_path = output_path / "merged_dataset"

    # =============== Step 1: Detect ===============
    print(
        f"Step 1/5: Detect '{prompt}' "
        f"in {len(image_urls)} images with Grounding DINO"
    )
    samples = collect_samples(
        image_urls=image_urls,
        objects=class_names,
        name_to_id=name_to_id,
    )

    # =============== Step 2: Log =================
    print(
        "\nStep 2/5: Log detections into "
        "YOLO and RF-DETR datasets"
    )
    create_datasets(
        samples=samples,
        categories=categories,
        yolo_path=yolo_path,
        rfdetr_path=rfdetr_path,
    )

    # =============== Step 3: Convert =============
    print(
        "\nStep 3/5: Convert YOLO dataset "
        "to RF-DETR/COCO"
    )
    convert_dataset(
        source_path=yolo_path,
        output_path=converted_path,
    )

    # =============== Step 4: Merge ===============
    print(
        "\nStep 4/5: Merge the RF-DETR/COCO datasets"
    )
    summary = merge_datasets(
        dataset_paths=[
            converted_path,
            rfdetr_path,
        ],
        output_path=merged_path,
    )
    print(f"  Classes: {summary['num_classes']}")
    print(f"  Splits:  {summary['counts']}")

    # =============== Step 5: Visualize ===============
    print("\nStep 5/5: Visualize the merged dataset")
    utils.visualize(merged_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the detect -> log -> convert -> "
            "merge -> visualize pipeline example."
        )
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/pipeline"),
        help=(
            "Root directory for pipeline outputs "
            "(default: results/pipeline)."
        ),
    )

    parser.add_argument(
        "--prompt",
        default="box, carton, person",
        help="Comma-separated object names.",
    )

    args = parser.parse_args()

    detection_pipeline_example(
        output_path=args.output_path,
        prompt=args.prompt,
        image_urls=IMAGE_URLS,
    )