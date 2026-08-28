"""Demonstrates filter_image_using_morphological_hitmiss operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def filter_image_using_morphological_hitmiss_example():
    """Applies filter_image_using_morphological_hitmiss operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/spanners_arranged.jpg"
    image = datatypes.Image.from_url(image_url).to_grayscale()
    
    segmented_image = cornea.segment_image_using_threshold(image=image)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_hitmiss(
        image=segmented_image,
        kernel_size=5,
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
