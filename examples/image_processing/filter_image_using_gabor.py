"""
Demonstrates filter_image_using_gabor operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_gabor_example():
    """Applies filter_image_using_gabor operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/finger_print.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_gabor(
        image=image,
        kernel_size=5,
        standard_deviation=5.0,
        orientation=90.0,
        wavelength=5.0,
        aspect_ratio=0.5,
        phase_offset=90.0,
        output_format="8bit",
    )

    logger.success(
        "Applied filter_image_using_gabor. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_gabor_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_gabor_example()
