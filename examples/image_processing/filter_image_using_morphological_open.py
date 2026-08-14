"""
Demonstrates morphological opening transformation.

This example:
- Downloads an example image.
- Applies open morphological operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_open_example():
    """Applies open morphological operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/broken_cables.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_open(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=2,
        border_type="constant",
        border_value=0,
    )

    logger.success(
        "Applied open morphological operation. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_open_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Opened")

if __name__ == "__main__":
    filter_image_using_morphological_open_example()
