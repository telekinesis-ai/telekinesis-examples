"""Demonstrates filter_image_using_morphological_hitmiss operation."""

import numpy as np
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
    # Fires at the centre of isolated pads about as large as the 35 px kernel: here the four
    # J1 connector pads. Larger blobs, IC leads and text are not matched.
    filtered_image = pupil.filter_image_using_morphological_hitmiss(
        image=segmented_image,
        kernel_size=35,
        kernel_shape="ellipse",
        iterations=1,
        border_type="constant",
        border_value=0,
    )

    # Hit-miss marks only the centre pixel(s) of each match; dilate them so they are visible.
    marked_image = pupil.filter_image_using_morphological_dilate(
        image=filtered_image, kernel_size=41, kernel_shape="ellipse", iterations=1,
    )
    marked_image = datatypes.Image((np.asarray(marked_image.data) > 0).astype(np.uint8) * 255)

    # ===================== Log ================================================
    logger.success(f"Applied filter_image_using_morphological_hitmiss on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_hitmiss_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(marked_image, entity_path="2-filtered")

if __name__ == "__main__":
    filter_image_using_morphological_hitmiss_example()
