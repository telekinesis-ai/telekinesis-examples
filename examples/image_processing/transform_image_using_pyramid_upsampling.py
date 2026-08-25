"""Demonstrates pyramid upsampling transformation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def transform_image_using_pyramid_upsampling_example():
    """Applies pyramid upsampling transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/buttons_arranged_downsampled.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_image_using_pyramid_upsampling(
        image=image,
        scale_factor=2.0,
    )
    filtered_image_1 = pupil.transform_image_using_pyramid_upsampling(
        image=filtered_image,
        scale_factor=2.0,
    )
    filtered_image_2 = pupil.transform_image_using_pyramid_upsampling(
        image=filtered_image_1,
        scale_factor=2.0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied pyramid upsampling on {image}")
    logger.success(f"Result: {filtered_image}, {filtered_image_1}, {filtered_image_2}")

    # ===================== Visualization  (Optional) ======================
    rr.init("transform_image_using_pyramid_upsampling_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-level1")
    datatypes.visualize(filtered_image_1, entity_path="3-level2")
    datatypes.visualize(filtered_image_2, entity_path="4-level3")

if __name__ == "__main__":
    transform_image_using_pyramid_upsampling_example()
