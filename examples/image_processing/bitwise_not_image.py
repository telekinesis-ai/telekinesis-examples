"""
Demonstrates bitwise NOT operation on an image.

This example:
- Downloads an example image.
- Performs bitwise NOT (inversion) operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def bitwise_not_image_example():
    """Performs bitwise NOT (inversion) on an image."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/einstein.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.bitwise_not_image(image=image)

    logger.success("Bitwise NOT. Output shape: {}", filtered_image.shape)

    # ===================== Visualization  (Optional) ======================
    rr.init("bitwise_not_image_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Inverted")

if __name__ == "__main__":
    bitwise_not_image_example()
