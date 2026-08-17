"""Demonstrates filter_image_using_meijering operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_meijering_example():
    """Applies filter_image_using_meijering operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/sidewalk_cracked.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_meijering(
        image=image,
        scale_start=1,
        scale_end=10,
        scale_step=2,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_meijering on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_meijering_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Filtered")

if __name__ == "__main__":
    filter_image_using_meijering_example()
