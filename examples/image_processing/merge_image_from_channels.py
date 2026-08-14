"""
Demonstrates merging color channels into an image.

This example:
- Downloads an example image.
- Splits the image into channels.
- Merges the channels back into an image.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def merge_image_from_channels_example():
    """Splits and merges image channels."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/fruits_carts.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    # Split image into channels
    image_channels = pupil.split_image_into_channels(image=image)

    # Convert ImageBatch to list[Image]
    channel_images = image_channels.to_list()

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_images),
    )

    # Merge channels back into an image
    filtered_image = pupil.merge_image_from_channels(channels=channel_images)

    logger.success(
        "Merged {} channels. Output image shape: {}",
        len(channel_images),
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("merge_image_from_channels_example", spawn=True)
    channel_names = ["Red", "Green", "Blue"]
    for i, channel_image in enumerate(channel_images):
        datatypes.visualize(channel_image, entity_path=f"{i + 1}-{channel_names[i]}")
    datatypes.visualize(filtered_image, entity_path="4-Merged")

if __name__ == "__main__":
    merge_image_from_channels_example()
