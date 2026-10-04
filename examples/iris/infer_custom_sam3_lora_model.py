"""Run object-detection inference using a deployed Iris model."""

from loguru import logger
import rerun as rr

from telekinesis import datatypes, iris_backend


def infer_custom_sam3_lora_model_example():
    """Run and visualize inference using a deployed SAM3 LoRA model."""
    # ===================== Load Image ======================================
    image = datatypes.Image.from_path(
        r"C:\Users\AmruthaVenkatesan\Documents\Telekinesis\Code"
        r"\telekinesis-iris\dataset\epson_dataset_real\test\000046.jpg"
    )
    objects = ["black_cable_gland", "multi_pin_circular_connector", "panel_mount_pilot_light", "50_pin_male_Centronics_connector"]

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
    logger.info(
        f"All detected object bounding boxes: {detection_results.bboxes}"
    )
    logger.info(f"All detected object scores: {detection_results.scores}")
    logger.info(
        "All detected object category IDs: "
        f"{detection_results.category_ids}"
    )

    if len(detection_results) > 0:
        # Indexed object is of type `COCOObjectDetectionResult`.
        first_detection = detection_results[0]
        logger.info(f"Detected object at index 0: {first_detection}")
        logger.info(
            f"Detected object at index 0 bounding box: {first_detection.bbox}"
        )
        logger.info(
            f"Detected object at index 0 score: {first_detection.score}"
        )
        logger.info(
            "Detected object at index 0 category ID: "
            f"{first_detection.category_id}"
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
    infer_custom_sam3_lora_model_example()
