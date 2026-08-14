"""
Demonstrates GrabCut segmentation.
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def segment_image_using_grab_cut_example():
    """Segments an image using the GrabCut algorithm."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/plastic_part.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    bbox = [220, 20, 930, 850]
    segmented_image = cornea.segment_image_using_grab_cut(
        image=image, 
        bbox=bbox, 
        num_iterations=2
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using the GrabCut algorithm.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_grab_cut_example", spawn=True)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


if __name__ == "__main__":
    segment_image_using_grab_cut_example()
