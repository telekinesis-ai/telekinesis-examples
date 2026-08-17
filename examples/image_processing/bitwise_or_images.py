"""Demonstrates bitwise OR operation between two images."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def bitwise_or_images_example():
    """Performs bitwise OR between two images."""
    # ===================== Load Images ==========================================
    image_url_a = "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    image_url_b = "https://assets.telekinesis.ai/examples/v1/images/rectangles_mask.png"
    image_a = datatypes.Image.from_url(image_url_a)
    image_b = datatypes.Image.from_url(image_url_b)

    # ===================== Resize Image B ==========================================
    image_b = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
        pad_color=(0, 0, 0),
    ).drop_alpha()
    logger.info(f"Resized {image_b} to match dimensions of {image_a}")

    # ===================== Run Skill ==========================================
    filtered_image = pupil.bitwise_or_images(image_a=image_a, image_b=image_b)

    # ===================== Log ================================================
    logger.success(f"Bitwise OR between {image_a} and {image_b}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("bitwise_or_images_example", spawn=True)
    datatypes.visualize(image_a, entity_path="1-Original")
    datatypes.visualize(image_b, entity_path="2-Resized")
    datatypes.visualize(filtered_image, entity_path="3-Filtered")

if __name__ == "__main__":
    bitwise_or_images_example()
