"""Demonstrates morphological erosion to shrink bright regions and remove small noise."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def filter_image_using_morphological_erode_example():
    """Applies erosion to shrink bright regions and remove small noise."""
    # ===================== Load Image ==========================================
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/xyzibd_flat_bars_scene36_gray.jpg"
    image = datatypes.Image.from_url(image_url)
    segmentation_data = cornea.segment_image_using_otsu_threshold(image=image).data
    image = datatypes.Image(segmentation_data)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_morphological_erode(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=6,
        border_type="default",
    )

    # ===================== Log ================================================
    logger.success(f"Applied erosion morphological operation on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_image_using_morphological_erode_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-eroded")

if __name__ == "__main__":
    filter_image_using_morphological_erode_example()
