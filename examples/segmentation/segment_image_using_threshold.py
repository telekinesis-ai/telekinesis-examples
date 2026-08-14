"""
Demonstrates basic threshold segmentation.
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def segment_image_using_threshold_example():
    """Applies a simple global threshold to segment the image."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts_scattered.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_using_threshold(
        image=image, min_value=45, max_value=255, threshold_type="binary"
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using a global threshold.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_threshold_example", spawn=True)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


if __name__ == "__main__":
    segment_image_using_threshold_example()
