"""Demonstrates filter_image_using_laplacian operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_laplacian_example():
    """Applies filter_image_using_laplacian operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/flat_mechanical_component_denoised.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_laplacian(
        image=image,
        output_format="32bit",
        kernel_size=5,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_laplacian on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_laplacian_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_laplacian_example()
