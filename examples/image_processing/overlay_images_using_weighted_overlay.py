"""Demonstrates weighted overlay blending of two images."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def overlay_images_using_weighted_overlay_example():
    """Blends two images using weighted overlay."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/rusted_metal_gear.jpg"
    image_a = datatypes.Image.from_url(image_url)
    image_a = pupil.resize_image_with_aspect_fit(
        image=image_a,
        resize_width=512,
        resize_height=512,
    )

    # ===================== Create Second Image ==========================================
    image_b = pupil.rotate_image(
        image=image_a, angle_in_deg=60.0, keep_image_size=True
    )

    # ===================== Run Skill ==========================================
    filtered_image = pupil.overlay_images_using_weighted_overlay(
        image_a=image_a,
        image_b=image_b,
        weight_a=0.5,
        weight_b=0.5,
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
