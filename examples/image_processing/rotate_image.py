"""
Demonstrates rotate_image operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def rotate_image_example():
    """Applies rotate_image operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/synthetic_data_bin.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.rotate_image(
        image=image,
        angle_in_deg=10,
        interpolation_method="linear",
        keep_image_size=True,
    )

    logger.success(
        "Applied rotate_image. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("rotate_image_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Rotated")

if __name__ == "__main__":
    rotate_image_example()
