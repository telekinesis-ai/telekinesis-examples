"""Demonstrates convert_image_color_space operation."""

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

    # ===================== Log ================================================
    logger.success(f"Applied convert_image_color_space on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("convert_image_color_space_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    convert_image_color_space_example()
