"""
Demonstrates foreground segmentation using BiRefNet.
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import cornea, datatypes

def segment_image_foreground_using_birefnet_example():
    """Segments the foreground from the background using BiRefNet."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/screws_standing.jpg"
    image = datatypes.Image.from_url(url=image_url).to_rgb()

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_foreground_using_birefnet(
        image=image, 
        mask_threshold=0
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using BiRefNet.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_foreground_using_birefnet_example", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(origin="/input_image", name="Input"),
        rrb.Spatial2DView(origin="/segmented_image", name="Output"),
    )
    rr.send_blueprint(blueprint)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


if __name__ == "__main__":
    segment_image_foreground_using_birefnet_example()
