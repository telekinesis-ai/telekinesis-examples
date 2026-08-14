"""
Demonstrates SLIC superpixel segmentation.
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def segment_image_using_slic_superpixel_example():
    """Segments an image into compact superpixels using the SLIC algorithm."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_using_slic_superpixel(
        image=image, num_segments=2, compactness=15.0, max_iterations=20,
        sigma=0.0, enforce_connectivity=True, start_label=1
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using the SLIC superpixel algorithm.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_slic_superpixel_example", spawn=True)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


if __name__ == "__main__":
    segment_image_using_slic_superpixel_example()
