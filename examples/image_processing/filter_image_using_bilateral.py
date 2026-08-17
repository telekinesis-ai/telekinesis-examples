"""Demonstrates filter_image_using_bilateral operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_bilateral_example():
    """Applies filter_image_using_bilateral operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts_scattered_noised.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_bilateral(
        image=image,
        neighborhood_diameter=5,
        spatial_sigma=75.0,
        color_intensity_sigma=100.0,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_bilateral on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_bilateral_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_bilateral_example()
