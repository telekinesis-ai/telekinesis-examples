"""Run object-detection inference using a deployed Iris model."""

import argparse
from pathlib import Path

from loguru import logger
import rerun as rr

from telekinesis import datatypes, iris_backend


def infer_custom_sam3_lora_model_remote_example(args: argparse.Namespace) -> None:
    """Run and visualize inference using a deployed SAM3 LoRA model."""
    # ===================== Load Image ======================================
    image = datatypes.Image.from_path(args.image)
    objects = [
        "black_cable_gland",
        "multi_pin_circular_connector",
        "panel_mount_pilot_light",
        "50_pin_male_Centronics_connector",
    ]

    # ===================== Run Skill ======================================
    detection_results, categories = iris_backend.infer(
        image=image,
        model_name="sam3_lora_epson_real",
        threshold=0.3,
        prompt=objects,
    )

    # ===================== Log ============================================
    logger.success(f"Detected objects in {image} using Iris.")
    logger.success(f"Results: {detection_results}")

    logger.info(f"Categories available: {categories}")
    logger.info(f"All detected object bounding boxes: {detection_results.bboxes}")
    logger.info(f"All detected object scores: {detection_results.scores}")
    logger.info(f"All detected object category IDs: {detection_results.category_ids}")

    if len(detection_results) > 0:
        # Indexed object is of type `COCOObjectDetectionResult`.
        first_detection = detection_results[0]
        logger.info(f"Detected object at index 0: {first_detection}")
        logger.info(f"Detected object at index 0 bounding box: {first_detection.bbox}")
        logger.info(f"Detected object at index 0 score: {first_detection.score}")
        logger.info(
            f"Detected object at index 0 category ID: {first_detection.category_id}"
        )
        category_names = dict(zip(categories.ids, categories.names, strict=True))
        logger.info(
            "Detected object at index 0 category name: "
            f"{category_names[first_detection.category_id]}"
        )

    # ===================== Visualization (Optional) =======================
    rr.init("infer_custom_sam3_lora_model_example", spawn=True)
    datatypes.visualize(image, entity_path="/image/")
    datatypes.visualize(
        detection_results,
        entity_path="/image/overlayed_detections",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to send for remote inference.",
    )
    args = parser.parse_args()
    infer_custom_sam3_lora_model_remote_example(args)
