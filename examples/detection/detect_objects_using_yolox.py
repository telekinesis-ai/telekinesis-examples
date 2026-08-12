"""
Detect objects using YOLOX.

Runs YOLOX object detection on an image and returns COCO-like annotations
with category names from the COCO 80-class label set.

The annotations and categories are used for visualization overlays.
"""

from loguru import logger
import rerun as rr

from telekinesis import retina, constants, datatypes

def detect_objects_using_yolox_example():
    """
    Detect objects using YOLOX.

    Runs YOLOX object detection on an image and returns object detections using datatype
    `COCOObjectDetectionResults`
    """
    # Load image
    image_url = "https://assets.telekinesis.ai/examples/v1/images/warehouse_2.jpg"
    image = datatypes.Image.from_url(url=image_url)
    logger.info(f"Loaded {image} from the URL: {image_url}")

    # Detect objects
    detection_results = retina.detect_objects_using_yolox(
        image=image,
        score_threshold=0.80,
        nms_threshold=0.45,
    )
    logger.info(f"YOLOX detected {len(detection_results)} object detections.")

    # Get COCO categories for YOLOX
    categories = constants.get_coco_categories(model="yolox")
    logger.info(f"YOLOX categories: {categories}")

    # Access the underlying grouped data
    all_bboxes = detection_results.bboxes
    all_scores = detection_results.scores
    all_category_ids = detection_results.category_ids
    logger.info(f"All detected object bounding boxes: {all_bboxes}")
    logger.info(f"All detected object scores: {all_scores}")
    logger.info(f"All detected object category IDs: {all_category_ids}")

    # Access individual detect objects at an index and log their details
    # Indexed objects are of type `COCOObjectDetectionAnnotation`
    index = 0
    detection_at_index = detection_results[index]
    detection_at_index_bbox = detection_at_index.bbox
    detection_at_index_score = detection_at_index.score
    detection_at_index_category_id = detection_at_index.category_id
    detection_at_index_category_name = categories[detection_at_index_category_id]

    logger.info(f"Detected object at index {index}: {detection_at_index}")
    logger.info(f"Detected object at index {index} bounding box: {detection_at_index_bbox}")
    logger.info(f"Detected object at index {index} score: {detection_at_index_score}")
    logger.info(f"Detected object at index {index} category ID: {detection_at_index_category_id}")
    logger.info(f"Detected object at index {index} category name: {detection_at_index_category_name}")

    # Visualize results using Rerun
    rr.init("detect_objects_using_yolox_example", spawn=True)
    datatypes.visualize(image, detection_results, categories, entity_path="/Image")


if __name__ == "__main__":
    detect_objects_using_yolox_example()
