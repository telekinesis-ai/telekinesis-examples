"""
Demonstrates pyramid downsampling transformation.

This example:
- Downloads an example image.
- Applies pyramid downsampling multiple times.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def transform_image_using_pyramid_downsampling_example():
    """Applies pyramid downsampling transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/gearbox.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_image_using_pyramid_downsampling(
        image=image,
        scale_factor=0.5,
    )
    filtered_image_1 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image,
        scale_factor=0.5,
    )
    filtered_image_2 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image_1,
        scale_factor=0.5,
    )

    logger.success(
        "Applied pyramid downsampling. Transformed output image shapes: {}, {}, {}",
        filtered_image.shape,
        filtered_image_1.shape,
        filtered_image_2.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("transform_image_using_pyramid_downsampling_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Level 1")
    datatypes.visualize(filtered_image_1, entity_path="3-Level 2")
    datatypes.visualize(filtered_image_2, entity_path="4-Level 3")

if __name__ == "__main__":
    transform_image_using_pyramid_downsampling_example()
