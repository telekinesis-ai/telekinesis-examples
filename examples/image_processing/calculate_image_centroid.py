"""
Demonstrates centroid calculation on a binary mask.

This example:
- Downloads an example binary image.
- Computes the centroid of non-zero pixels.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def calculate_image_centroid_example():
    """Computes the centroid of a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/metal_part_mask.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    centroid = pupil.calculate_image_centroid(mask=image)

    logger.success("Computed centroid. Position: {}", centroid.data)

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_image_centroid_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Mask")
    datatypes.visualize(centroid, entity_path="2-Centroid")

if __name__ == "__main__":
    calculate_image_centroid_example()
