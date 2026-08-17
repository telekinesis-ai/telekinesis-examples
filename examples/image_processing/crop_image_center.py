"""Demonstrates crop_image_center operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_center_example():
    """Applies crop_image_center operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/rusted_metal_gear.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.crop_image_center(
        image=image,
        crop_width=300,
        crop_height=300,
        pad_color=(0, 0, 0),
    )

    # ===================== Log ================================================
    logger.success(f"Applied crop_image_center on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("crop_image_center_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Cropped")

if __name__ == "__main__":
    crop_image_center_example()
