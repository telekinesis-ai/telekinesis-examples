"""
Demonstrates LAB color space segmentation.
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import cornea, datatypes

def segment_image_using_lab_example():
    """Segments an image using LAB color space range."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/car_painting.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_using_lab(
        image=image, lower_bound=(120, 50, 50), upper_bound=(180, 255, 255)
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using LAB color space range.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_lab_example", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(origin="/input_image", name="Input"),
        rrb.Spatial2DView(origin="/segmented_image", name="Output"),
    )
    rr.send_blueprint(blueprint)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


if __name__ == "__main__":
    segment_image_using_lab_example()
