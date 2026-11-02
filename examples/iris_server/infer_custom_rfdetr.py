"""Run RF-DETR inference using a model deployed to Iris Server."""

import argparse
from pathlib import Path

from loguru import logger
import rerun as rr

from telekinesis import datatypes, iris_backend


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to send for remote inference.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the deployed RF-DETR model.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def infer_rfdetr(args: argparse.Namespace) -> None:
    """Run and optionally visualize remote RF-DETR inference."""
    image = datatypes.Image.from_path(args.image)
    detection_results, categories = iris_backend.infer(
        image=image,
        model_name=args.model_name,
        threshold=args.threshold,
    )

    logger.success(f"Detected objects in {image} using Iris.")
    logger.success(f"Results: {detection_results}")
    logger.info(f"Categories available: {categories}")
    logger.info(f"All detected object bounding boxes: {detection_results.bboxes}")
    logger.info(f"All detected object scores: {detection_results.scores}")
    logger.info(f"All detected object category IDs: {detection_results.category_ids}")

    if len(detection_results) > 0:
        first_detection = detection_results[0]
        category_names = dict(zip(categories.ids, categories.names, strict=True))
        logger.info(f"Detected object at index 0: {first_detection}")
        logger.info(f"Detected object at index 0 bounding box: {first_detection.bbox}")
        logger.info(f"Detected object at index 0 score: {first_detection.score}")
        logger.info(
            f"Detected object at index 0 category ID: {first_detection.category_id}"
        )
        logger.info(
            "Detected object at index 0 category name: "
            f"{category_names[first_detection.category_id]}"
        )

    if args.visualize:
        rr.init("infer_rfdetr", spawn=True)
        datatypes.visualize(image, entity_path="/image/")
        datatypes.visualize(
            detection_results,
            entity_path="/image/overlayed_detections",
        )


if __name__ == "__main__":
    infer_rfdetr(parse_args())
