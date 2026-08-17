"""Demonstrates merging color channels into an image."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def merge_image_from_channels_example():
    """Splits and merges image channels."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/fruits_carts.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    image_channels = pupil.split_image_into_channels(image=image)
    filtered_image = pupil.merge_image_from_channels(channels=image_channels)

    # ===================== Log ================================================
    logger.success(f"Split and merged channels of {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("merge_image_from_channels_example", spawn=True)
    channel_names = ["Red", "Green", "Blue"]
    for i, channel_image in enumerate(image_channels):
        datatypes.visualize(channel_image, entity_path=f"{i + 1}-{channel_names[i]}")
    datatypes.visualize(filtered_image, entity_path="4-Merged")

if __name__ == "__main__":
    merge_image_from_channels_example()
