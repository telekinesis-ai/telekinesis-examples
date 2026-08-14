"""
Demonstrates filter_image_using_frangi operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_frangi_example():
    """Applies filter_image_using_frangi operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/tablets_arranged.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_frangi(
        image=image,
        scale_start=6,
        scale_end=10,
        scale_step=1,
        alpha=0.5,
        beta=0.5,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    logger.success(
        "Applied filter_image_using_frangi. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_frangi_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_frangi_example()
