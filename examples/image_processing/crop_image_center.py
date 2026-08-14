"""
Demonstrates crop_image_center operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_center_example():
    """Applies crop_image_center operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/rusted_metal_gear.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.crop_image_center(
        image=image,
        crop_width=300,
        crop_height=300,
        pad_color=(0, 0, 0),
    )

    logger.success(
        "Applied crop_image_center. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("crop_image_center_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Cropped")

if __name__ == "__main__":
    crop_image_center_example()
