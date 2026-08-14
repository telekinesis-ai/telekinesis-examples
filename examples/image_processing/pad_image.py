"""
Demonstrates pad_image operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def pad_image_example():
    """Applies pad_image operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/bin_picking_metal_2.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.pad_image(
        image=image,
        top=200,
        bottom=50,
        left=100,
        right=75,
        border_type="constant",
        border_value=0.0,
    )

    logger.success(
        "Applied pad_image. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("pad_image_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Padded")

if __name__ == "__main__":
    pad_image_example()
