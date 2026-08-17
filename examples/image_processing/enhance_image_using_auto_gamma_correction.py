"""Demonstrates enhance_image_using_auto_gamma_correction operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def enhance_image_using_auto_gamma_correction_example():
    """Applies enhance_image_using_auto_gamma_correction operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/screws_in_dark_lighting.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.enhance_image_using_auto_gamma_correction(
        image=image,
    )

    # ===================== Log ================================================
    logger.success(f"Applied enhance_image_using_auto_gamma_correction on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("enhance_image_using_auto_gamma_correction_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Enhanced")

if __name__ == "__main__":
    enhance_image_using_auto_gamma_correction_example()
