"""Demonstrates splitting an image into color channels."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def split_image_into_channels_example():
    """Splits an image into its color channels."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/vegetables.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    image_channels = pupil.split_image_into_channels(image=image)

    # ===================== Log ================================================
    logger.success(f"Split {image} into channels")
    logger.success(f"Result: {image_channels}")

    # ===================== Visualization  (Optional) ======================
    rr.init("split_image_into_channels_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    channel_names = ["Red", "Green", "Blue"]
    for i, channel_image in enumerate(image_channels):
        datatypes.visualize(channel_image, entity_path=f"{i + 2}-{channel_names[i]}")

if __name__ == "__main__":
    split_image_into_channels_example()
