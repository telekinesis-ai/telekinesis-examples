"""Demonstrates centroid calculation on a binary mask."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def calculate_mask_centroid_example():
    """Computes the centroid of a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/metal_part_mask.png"
    image = datatypes.Image.from_url(image_url)
    mask = cornea.segment_image_using_otsu_threshold(image=image)

    # ===================== Run Skill ==========================================
    centroid = pupil.calculate_mask_centroid(mask=mask)

    # ===================== Log ================================================
    logger.success(f"Calculated centroid of {image}")
    logger.success(f"Result: {centroid}")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_mask_centroid_example", spawn=True)
    datatypes.visualize(mask, centroid, entity_path="/masked_image", label="Centroid")

if __name__ == "__main__":
    calculate_mask_centroid_example()
