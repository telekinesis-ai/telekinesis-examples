"""
Demonstrates convert_image_color_space operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def convert_image_color_space_example():
    """Applies convert_image_color_space operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/apples_black_container.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.convert_image_color_space(
        image=image,
    source_color_space="RGB",
    target_color_space="GRAY",
    )

    logger.success(
        "Applied convert_image_color_space. Output image shape: {}", 
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("convert_image_color_space_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    convert_image_color_space_example()
