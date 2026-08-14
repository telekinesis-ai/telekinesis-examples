"""
Demonstrates resize_image_with_aspect_fit operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def resize_image_with_aspect_fit_example():
    """Applies resize_image_with_aspect_fit operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/gearbox.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.resize_image_with_aspect_fit(
        image=image,
        resize_width=400,
        resize_height=300,
        interpolation_method="linear",
    )

    logger.success(
        "Applied resize_image_with_aspect_fit. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("resize_image_with_aspect_fit_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Resized")

if __name__ == "__main__":
    resize_image_with_aspect_fit_example()
