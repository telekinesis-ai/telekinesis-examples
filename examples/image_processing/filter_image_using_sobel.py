"""Demonstrates filter_image_using_sobel operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_sobel_example():
    """Applies filter_image_using_sobel operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_sobel(
        image=image,
        output_format="64bit",
        dx=1,
        dy=1,
        kernel_size=9,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_sobel on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_sobel_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_sobel_example()
