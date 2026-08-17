"""Demonstrates rotate_image operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def rotate_image_example():
    """Applies rotate_image operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/synthetic_data_bin.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.rotate_image(
        image=image,
        angle_in_deg=10,
        interpolation_method="linear",
        keep_image_size=True,
    )

    # ===================== Log ================================================
    logger.success(f"Applied rotate_image on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("rotate_image_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Rotated")

if __name__ == "__main__":
    rotate_image_example()
