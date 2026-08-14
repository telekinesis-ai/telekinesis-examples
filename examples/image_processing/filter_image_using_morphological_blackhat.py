"""
Demonstrates filter_image_using_morphological_blackhat operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

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

    logger.success(
        "Applied filter_image_using_morphological_blackhat. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_blackhat_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_morphological_blackhat_example()
