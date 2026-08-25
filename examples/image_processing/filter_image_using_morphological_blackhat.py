"""Demonstrates filter_image_using_morphological_blackhat operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_blackhat_example():
    """Applies filter_image_using_morphological_blackhat operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/mechanical_parts_gray.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_blackhat(
        image=image,
        kernel_size=15,
        kernel_shape="ellipse",
        iterations=2,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_morphological_blackhat on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_blackhat_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_morphological_blackhat_example()
