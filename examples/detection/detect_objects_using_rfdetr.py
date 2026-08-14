"""
Detect objects using RF-DETR.

Runs RF-DETR object detection on an image and returns COCO-like annotations
with category names from the COCO 80-class label set.

The annotations and categories are used for visualization overlays.
"""

from loguru import logger
import rerun as rr

from telekinesis import retina, constants, datatypes


def detect_objects_using_rfdetr_example():
    """
    Detect objects using RF-DETR.

    Runs RF-DETR object detection on an image and returns object detections using datatype
    `COCOObjectDetectionResults`.
    """
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/warehouse_1.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    detection_results = retina.detect_objects_using_rfdetr(
        image=image,
        score_threshold=0.5,
    )

    # ===================== Log ================================================
    logger.success(f"Detected objects in {image} using RF-DETR.")
    logger.success(f"Results: {detection_results}")

    categories = constants.get_coco_categories(model="rfdetr")
    logger.info(f"RF-DETR categories: {categories}")

    logger.info(f"All detected object bounding boxes: {detection_results.bboxes}")
    logger.info(f"All detected object scores: {detection_results.scores}")
    logger.info(f"All detected object category IDs: {detection_results.category_ids}")

    # Indexed object is of type the single `COCOObjectDetectionResult`
    logger.info(f"Detected object at index 0: {detection_results[0]}")
    logger.info(f"Detected object at index 0 bounding box: {detection_results[0].bbox}")
    logger.info(f"Detected object at index 0 score: {detection_results[0].score}")
    logger.info(
        f"Detected object at index 0 category ID: {detection_results[0].category_id}"
    )
    logger.info(
        f"Detected object at index 0 category name: {categories[detection_results[0].category_id]}"
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("detect_objects_using_rfdetr_example", spawn=True)
    datatypes.visualize(image, entity_path="/image")
    datatypes.visualize(detection_results, entity_path="/image/overlayed_detections")


if __name__ == "__main__":
    detect_objects_using_rfdetr_example()
