"""Demonstrates filter_image_using_blur operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_blur_example():
    """Applies filter_image_using_blur operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts_scattered_noised.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_blur(
        image=image,
        kernel_size=7,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_blur on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_blur_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_blur_example()
