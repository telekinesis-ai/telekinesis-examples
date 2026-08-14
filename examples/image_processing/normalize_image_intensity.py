"""
Demonstrates normalize_image_intensity operation.

This example:
- Downloads an example image.
- Applies the operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def normalize_image_intensity_example():
    """Applies normalize_image_intensity operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/gauge_washed.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.normalize_image_intensity(
        image=image,
        alpha=0.0,
        beta=255.0,
        normalization_method="minmax",
        output_format="8bit",
    )

    logger.success(
        "Applied normalize_image_intensity. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("normalize_image_intensity_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Normalized")

if __name__ == "__main__":
    normalize_image_intensity_example()
