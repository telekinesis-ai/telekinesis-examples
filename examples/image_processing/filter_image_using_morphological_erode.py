"""
Demonstrates morphological erosion to shrink bright regions and remove small noise.

This example:
- Downloads an example image.
- Applies erosion morphological operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_erode_example():
    """Applies erosion to shrink bright regions and remove small noise."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/gear_with_texture.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_erode(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=10,
        border_type="default",
    )

    logger.success(
        "Applied erosion morphological operation. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_erode_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Eroded")

if __name__ == "__main__":
    filter_image_using_morphological_erode_example()
