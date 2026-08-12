"""
Detect objects using Grounding DINO (zero-shot).

Uses a free-form text prompt to detect objects in an RGB image.
Returns COCO-like annotations with bounding boxes.

The annotations and categories are used for visualization overlays.
"""


from loguru import logger
import rerun as rr

from telekinesis import retina, datatypes


def detect_objects_using_grounding_dino_example():
    """
    Detect objects using Grounding DINO (zero-shot).

    Requires the input objects to be defined to detect objects in an image and
    returns object detections using datatype `COCOObjectDetectionDetectionResults` with `Categories`.
    """
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/palletizing.jpg"
    image = datatypes.Image.from_url(url=image_url)
    logger.info(f"Loaded {image} from the URL: {image_url}")

    # ===================== Run Skill ==========================================
    detection_results, categories = retina.detect_objects_using_grounding_dino(
        image=image,
        objects=["carton"],
        box_threshold=0.5,
        text_threshold=0.5,
    )
    logger.info(f"Grounding DINO detected {len(detection_results)} objects.")
    logger.info(f"Categories available: {categories}")

    # Access the first detected object, which is COCOObjectDetectionAnnotation
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

    # ===================== Visualization  (Optional) ======================
    rr.init("detect_objects_using_grounding_dino_example", spawn=True)
    datatypes.visualize(image, detection_results, categories, entity_path="/Image")


if __name__ == "__main__":
    detect_objects_using_grounding_dino_example()
