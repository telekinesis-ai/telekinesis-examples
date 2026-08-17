"""Demonstrates filter_image_using_morphological_gradient operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_gradient_example():
    """Applies filter_image_using_morphological_gradient operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/cartons_arranged.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_gradient(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=1,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_morphological_gradient on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_gradient_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_morphological_gradient_example()
