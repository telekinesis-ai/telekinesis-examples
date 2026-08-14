"""
Demonstrates filter_image_using_morphological_hitmiss operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_hitmiss_example():
    """Applies filter_image_using_morphological_hitmiss operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/spanners_arranged.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_hitmiss(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=1,
        border_type="constant",
        border_value=0,
    )

    logger.success(
        "Applied filter_image_using_morphological_hitmiss. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_hitmiss_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_morphological_hitmiss_example()
