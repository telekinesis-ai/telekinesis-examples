"""
Demonstrates filtering superpixels based on color.
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import cornea, datatypes

def filter_segments_by_color_example():
    """Filters superpixels based on color criteria."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/eggs_carton.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    superpixel_segmentation_image = cornea.segment_image_using_felzenszwalb(
        image=image, scale=500, sigma=1, min_size=200
    )
    filtered_image = cornea.filter_segments_by_color(
        image=image, labels=superpixel_segmentation_image,
        min_color=0, max_color=125.0
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {image} superpixels by color.")
    logger.success(f"Results: {filtered_image}")
    logger.info(f"Filtered image label codes: {filtered_image.label_codes}")
    logger.info(f"Filtered image number of labels: {filtered_image.number_of_labels}")
    logger.info(f"Filtered image shape: {filtered_image.shape}")
    logger.info(f"Filtered image dtype: {filtered_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_segments_by_color_example", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(origin="/input_image", name="Input"),
        rrb.Spatial2DView(origin="/filtered_image", name="Output"),
    )
    rr.send_blueprint(blueprint)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(filtered_image, entity_path="/filtered_image")


if __name__ == "__main__":
    filter_segments_by_color_example()
