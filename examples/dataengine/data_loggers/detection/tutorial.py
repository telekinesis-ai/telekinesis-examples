"""
End-to-end usage flow: detect -> log -> convert -> merge -> visualize.

Runs the full pipeline in one script so you can see how the other examples in
this directory connect. Each stage below has a standalone counterpart with its
own CLI flags:

  1. Detect objects in real images with Grounding DINO (zero-shot, from a text
     prompt) -- see detect_objects_using_grounding_dino.py.
  2. Log two datasets from those detections, one YOLO and one RF-DETR -- see
     logger_inline.py.
  3. Convert the YOLO dataset to RF-DETR/COCO -- see convert.py.
  4. Merge both RF-DETR/COCO datasets into one -- see merge.py.
  5. Visualize the merged result -- see visualize.py. Launches the FiftyOne
     app and blocks until you close it.

Everything is written under --output-path, which is wiped and rebuilt on every
run so the pipeline is safe to re-run as-is.

Note: this needs both the ``telekinesis`` skills package (for ``retina`` /
``pupil`` / ``datatypes``) and ``telekinesis.dataengine`` importable in the
same environment, plus network access to the Grounding DINO backend and to
the image URLs below.

Usage:
    python tutorial.py
    python tutorial.py --prompt "forklift . pallet ."
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from loguru import logger

from telekinesis import retina, datatypes
from telekinesis.dataengine import DetectionLogger
from telekinesis.dataengine.detection.utils import convert_dataset, merge_datasets, visualize

# The images this tutorial detects objects in. Swap these for your own URLs
# (or local files) to try the pipeline on different scenes.
IMAGE_URLS = [
    "https://assets.telekinesis.ai/examples/v1/images/warehouse_1.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/warehouse_2.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/palletizing.jpg",
    "https://assets.telekinesis.ai/examples/v1/images/cartons_arranged.png",
    "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg",
]


def detect_sample(
    image: datatypes.Image,
    class_names: list[str],
    name_to_id: dict[str, int],
) -> tuple[np.ndarray, list[dict]]:
    """Run Grounding DINO on one image; return it with COCO-style annotations.

    Grounding DINO assigns category ids per call, in the order classes first
    appear, so two images can disagree on what id means what class. Remapping
    by name onto ``name_to_id`` (built once, up front) keeps ids consistent
    across every logged image -- and across the two datasets merged later.
    """
    annotations, categories = retina.detect_objects_using_grounding_dino(
        image=image,
        objects=class_names,
        box_threshold=0.3,
        text_threshold=0.25,
    )
    local_names = dict(zip(categories.ids.tolist(), categories.names.tolist()))

    remapped = []
    for category_id, bbox in zip(
        annotations.category_ids.tolist(), annotations.bboxes.tolist()
    ):
        name = local_names.get(category_id)
        class_id = name_to_id.get(name)
        if class_id is None:
            logger.warning(f"unexpected class {name!r} from Grounding DINO -- skipping")
            continue
        remapped.append({"category_id": class_id, "bbox": bbox})

    return image.to_numpy(), remapped


def log_dataset(
    output_format: str,
    output_path: pathlib.Path,
    categories: list[dict],
    samples: list[tuple[np.ndarray, list[dict]]],
) -> None:
    dataset_logger = DetectionLogger.create(
        output_format, output_path, categories, mode="overwrite"
    )
    for image, annotations in samples:
        dataset_logger.log(image, annotations)
    dataset_logger.close()


def main(output_path: pathlib.Path, prompt: str, image_urls: list[str]) -> None:
    class_names = [name.strip() for name in prompt.split(".") if name.strip()]
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]
    name_to_id = {c["name"]: c["id"] for c in categories}

    yolo_dir = output_path / "yolo_dataset"
    rfdetr_dir = output_path / "rfdetr_dataset"
    converted_dir = output_path / "yolo_converted"
    merged_dir = output_path / "merged_dataset"

    print(f"Step 1/5: detect '{prompt}' in {len(image_urls)} images with Grounding DINO")
    samples = []
    for url in image_urls:
        image = datatypes.Image.from_url(url)
        sample = detect_sample(image, class_names, name_to_id)
        samples.append(sample)
        logger.success(f"detected {len(sample[1])} objects in {url}")

    # Split the images between the two datasets rather than logging the same
    # ones into both -- that way the merge step in step 4 is actually
    # combining two distinct datasets, not one dataset with itself.
    split = len(samples) // 2
    yolo_samples, rfdetr_samples = samples[:split], samples[split:]

    print("\nStep 2/5: log two datasets from those detections (one YOLO, one RF-DETR)")
    log_dataset("yolo", yolo_dir, categories, yolo_samples)
    log_dataset("rfdetr", rfdetr_dir, categories, rfdetr_samples)

    print("\nStep 3/5: convert the YOLO dataset to RF-DETR/COCO")
    convert_dataset(yolo_dir, converted_dir, "rfdetr", overwrite=True)

    print("\nStep 4/5: merge the two RF-DETR/COCO datasets")
    summary = merge_datasets([converted_dir, rfdetr_dir], merged_dir, overwrite=True)
    print(f"  classes: {summary['num_classes']}")
    print(f"  splits : {summary['counts']}")

    print("\nStep 5/5: visualize the merged result")
    visualize(merged_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the detect -> log -> convert -> merge -> visualize pipeline."
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default="results/pipeline",
        help="Root directory for every stage's output (default: results/pipeline)",
    )
    parser.add_argument(
        "--prompt",
        default="box . carton . person .",
        help="Dot-separated Grounding DINO text prompt (default: %(default)r)",
    )
    args = parser.parse_args()

    main(output_path=args.output_path, prompt=args.prompt, image_urls=IMAGE_URLS)
