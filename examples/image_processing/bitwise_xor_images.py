"""Demonstrates bitwise XOR operation between two images."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def bitwise_xor_images_example():
    """Performs bitwise XOR between two images."""
    # ===================== Load Images ==========================================
    image_url_a = "https://assets.telekinesis.ai/examples/v1/images/image_1.png"
    image_url_b = "https://assets.telekinesis.ai/examples/v1/images/image_2.png"
    image_a = datatypes.Image.from_url(image_url_a)
    image_b = datatypes.Image.from_url(image_url_b)

    # ===================== Resize Image B ==========================================
    image_b_resized = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
    )

    # ===================== Run Skill ==========================================
    filtered_image = pupil.bitwise_xor_images(image_a=image_a, image_b=image_b_resized)

    # ===================== Log ================================================
    logger.success(f"Bitwise XOR between {image_a} and {image_b_resized}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("bitwise_xor_images_example", spawn=True)
    datatypes.visualize(image_a, entity_path="1-original")
    datatypes.visualize(image_b_resized, entity_path="2-resized")
    datatypes.visualize(filtered_image, entity_path="3-filtered")

if __name__ == "__main__":
    bitwise_xor_images_example()
