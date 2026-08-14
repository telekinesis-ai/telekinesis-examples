"""
Demonstrates enhance_image_using_white_balance operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def enhance_image_using_white_balance_example():
    """Applies enhance_image_using_white_balance operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/hand_tools_yellow_light.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.enhance_image_using_white_balance(
        image=image,
    )

    logger.success(
        "Applied enhance_image_using_white_balance. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("enhance_image_using_white_balance_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Enhanced")

if __name__ == "__main__":
    enhance_image_using_white_balance_example()
