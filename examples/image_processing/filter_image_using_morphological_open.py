"""Demonstrates morphological opening transformation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_morphological_open_example():
    """Applies open morphological operation."""
    # ===================== Load Image ==========================================
    # Example data: FabriD (CC BY 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/fabrid_black_knit_lint_threshold_mask.png"
    image = datatypes.Image.from_url(image_url).to_grayscale()

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_open(
        image=image,
        kernel_size=7,
        kernel_shape="ellipse",
        iterations=1,
        border_type="constant",
        border_value=0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied open morphological operation on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_open_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-opened")

if __name__ == "__main__":
    filter_image_using_morphological_open_example()
