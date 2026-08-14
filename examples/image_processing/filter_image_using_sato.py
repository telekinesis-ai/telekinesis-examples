"""
Demonstrates filter_image_using_sato operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_sato_example():
    """Applies filter_image_using_sato operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pcb_top_gray.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_sato(
        image=image,
        scale_start=1,
        scale_end=12,
        scale_step=1,
        detect_black_ridges=False,
        border_type="reflect",
        border_value=0.0,
    )

    logger.success(
        "Applied filter_image_using_sato. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_sato_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_sato_example()
