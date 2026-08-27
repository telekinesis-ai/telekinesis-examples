"""Demonstrates morphological dilation to expand bright regions and fill holes."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_dilate_example():
    """Applies dilation to expand bright regions and fill holes."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/spanners_arranged.jpg"
    image = datatypes.Image.from_url(image_url).to_grayscale()

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_dilate(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="constant",
        border_value=0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied dilation morphological operation on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_dilate_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-dilated")

if __name__ == "__main__":
    filter_image_using_morphological_dilate_example()
