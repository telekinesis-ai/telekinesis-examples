"""
Demonstrates splitting an image into color channels.

This example:
- Downloads an example image.
- Splits the image into its color channels.
- Visualizes each channel using Rerun.
"""

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

    # Convert ImageBatch to list[Image]
    channel_images = image_channels.to_list()

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_images),
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("split_image_into_channels_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    channel_names = ["Red", "Green", "Blue"]
    for i, channel_image in enumerate(channel_images):
        datatypes.visualize(channel_image, entity_path=f"{i + 2}-{channel_names[i]}")

if __name__ == "__main__":
    split_image_into_channels_example()
