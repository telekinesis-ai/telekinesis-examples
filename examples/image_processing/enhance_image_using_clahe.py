"""Demonstrates enhance_image_using_clahe operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def enhance_image_using_clahe_example():
    """Applies enhance_image_using_clahe operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/dark_warehouse.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.enhance_image_using_clahe(
        image=image,
        clip_limit=10.0,
        tile_grid_size=8,
        color_space="lab",
    )

    # ===================== Log ================================================
    logger.success(f"Applied enhance_image_using_clahe on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("enhance_image_using_clahe_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Enhanced")

if __name__ == "__main__":
    enhance_image_using_clahe_example()
