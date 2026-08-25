"""Demonstrates filter_image_using_hessian operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def filter_image_using_hessian_example():
    """Applies filter_image_using_hessian operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/wires.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_hessian(
        image=image,
        scale_start=1,
        scale_end=6,
        scale_step=1,
        alpha=0.5,
        beta=0.5,
        gamma=15,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_hessian on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_hessian_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_hessian_example()
