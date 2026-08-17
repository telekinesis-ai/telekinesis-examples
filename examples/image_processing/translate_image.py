"""Demonstrates translate_image operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def translate_image_example():
    """Applies translate_image operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/checkerboard.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.translate_image(
        image=image,
        dx=100,
        dy=50,
        border_type="constant",
        border_value=0,
        interpolation_method="linear",
    )

    # ===================== Log ================================================
    logger.success(f"Applied translate_image on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("translate_image_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Translated")

if __name__ == "__main__":
    translate_image_example()
