"""
Demonstrates filtering superpixels by area.
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def filter_segments_by_area_example():
    """Filters superpixels based on area."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/eggs_carton.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    superpixel_segmentation_image = cornea.segment_image_using_felzenszwalb(
        image=image, 
        scale=500,
        sigma=1,
        min_size=200
    )

    filtered_image = cornea.filter_segments_by_area(
        image=image, labels=superpixel_segmentation_image,
        min_area=10000, max_area=100000
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {image} superpixels by area.")
    logger.success(f"Results: {filtered_image}")
    logger.info(f"Filtered image label codes: {filtered_image.label_codes}")
    logger.info(f"Filtered image number of labels: {filtered_image.number_of_labels}")
    logger.info(f"Filtered image shape: {filtered_image.shape}")
    logger.info(f"Filtered image dtype: {filtered_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_segments_by_area_example", spawn=True)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(filtered_image, entity_path="/filtered_image")


if __name__ == "__main__":
    filter_segments_by_area_example()
