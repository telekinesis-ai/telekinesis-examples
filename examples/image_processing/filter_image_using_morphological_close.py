"""Demonstrates morphological closing to fill small holes and close gaps."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_close_example():
    """Applies close morphological operation to fill holes and close gaps."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts_scattered.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_close(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied close morphological operation on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_close_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-closed")

if __name__ == "__main__":
    filter_image_using_morphological_close_example()
