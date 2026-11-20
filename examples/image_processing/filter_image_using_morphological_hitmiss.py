"""Demonstrates filter_image_using_morphological_hitmiss operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def filter_image_using_morphological_hitmiss_example():
    """Applies filter_image_using_morphological_hitmiss operation."""
    # ===================== Load Image ==========================================
    # Example data: VisA (CC BY 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/visa_pcb_driver_board_layout_dark.png"
    image = datatypes.Image.from_url(image_url).to_grayscale()

    # Pads and silkscreen are dark in this layout: invert so they become the foreground.
    copper_image = pupil.bitwise_not_image(image=image)
    segmented_image = cornea.segment_image_using_threshold(image=copper_image)

    # ===================== Run Skill ==========================================
    # Fires only where a 21 px disc fits inside a feature: SMD pads and vias, not IC leads or text.
    filtered_image = pupil.filter_image_using_morphological_hitmiss(
        image=segmented_image,
        kernel_size=21,
        kernel_shape="ellipse",
        iterations=1,
        border_type="constant",
        border_value=0,
    )

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_morphological_hitmiss on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_hitmiss_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_morphological_hitmiss_example()
