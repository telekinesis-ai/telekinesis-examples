"""
Demonstrates merging color channels into an image.

This example:
- Downloads an example image.
- Splits the image into channels.
- Merges the channels back into an image.
- Visualizes the result using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from datatypes import datatypes
from telekinesis import pupil


def merge_image_from_channels_example():
    """Splits and merges image channels."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/fruits_carts.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    # Split image into channels
    image_channels = pupil.split_image_into_channels(image=image)

    # Convert ListOfImages → list[Image]
    channel_images = image_channels.to_list()

    # Convert channels to numpy
    channel_np_list = [img.to_numpy() for img in channel_images]

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_np_list),
    )

    # Merge channels back into an image
    merged_image = pupil.merge_image_from_channels(channels=channel_images)

    merged_image_np = merged_image.to_numpy()

    logger.success(
        "Merged {} channels. Output image shape: {}",
        len(channel_images),
        merged_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(channel_np_list, merged_image_np)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads an image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    image = datatypes.Image(image=image_bgr, color_model="BGR")
    image = pupil.convert_image_color_space(
        image, source_color_space="BGR", target_color_space="RGB"
    )
    logger.success(f"Loaded image from {image_url}")
    return image


def visualize(channel_np_list, merged_image_np) -> None:
    """Visualizes the channels and merged image using Rerun."""
    rr.init("merge_image_from_channels", spawn=True)

    # Create Spatial2DViews dynamically
    views = []

    channel_names = ["Red", "Green", "Blue"]

    for i in range(len(channel_np_list)):
        views.append(
            rrb.Spatial2DView(
                name=channel_names[i],
                origin=f"channel_{i + 1}",
            )
        )

    views.append(
        rrb.Spatial2DView(
            name="Merged",
            origin="merged_image",
        )
    )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(*views),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log channel images
    for i, channel_np in enumerate(channel_np_list):
        rr.log(f"channel_{i + 1}", rr.Image(channel_np))

    # Log merged image
    rr.log("merged_image", rr.Image(merged_image_np))


if __name__ == "__main__":
    merge_image_from_channels_example()
