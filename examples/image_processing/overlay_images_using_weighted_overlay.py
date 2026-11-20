"""Demonstrates weighted overlay blending of two images."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def overlay_images_using_weighted_overlay_example():
    """Blends two images using weighted overlay."""
    # ===================== Load Image ==========================================
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/xyzibd_gear_bin_photoneo.jpg"
    image_a = datatypes.Image.from_url(image_url)

    # ===================== Load Second Image (colorized instance mask) ==========================================
    mask_url = "https://assets.telekinesis.ai/examples/v1/images/xyzibd_gear_bin_instances.png"
    image_b = datatypes.Image.from_url(mask_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.overlay_images_using_weighted_overlay(
        image_a=image_a,
        image_b=image_b,
        weight_a=0.6,
        weight_b=0.4,
    )

    # ===================== Log ================================================
    logger.success(f"Weighted overlay between {image_a} and {image_b}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("overlay_images_using_weighted_overlay_example", spawn=True)
    datatypes.visualize(image_a, entity_path="1-imagea")
    datatypes.visualize(image_b, entity_path="2-imageb")
    datatypes.visualize(filtered_image, entity_path="3-blended")

if __name__ == "__main__":
    overlay_images_using_weighted_overlay_example()
