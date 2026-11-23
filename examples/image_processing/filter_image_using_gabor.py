"""Demonstrates filter_image_using_gabor operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_gabor_example():
    """Applies filter_image_using_gabor operation."""
    # ===================== Load Image ==========================================
    # Example data: FabriD (CC BY 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/fabrid_black_knit_missing_yarn.jpg"
    image = datatypes.Image.from_url(image_url).to_grayscale()

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_gabor(
        image=image,
        kernel_size=61,
        standard_deviation=10.0,
        orientation=93.4,
        wavelength=40.0,
        aspect_ratio=0.3,
        phase_offset=0.0,
        output_format="32bit",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_gabor on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_gabor_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_gabor_example()
