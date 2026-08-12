"""
Demonstrates splitting an image into color channels.

This example:
- Downloads an example image.
- Splits the image into its color channels.
- Visualizes each channel using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from datatypes import datatypes
from telekinesis import pupil


def split_image_into_channels_example():
    """Splits an image into its color channels."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/vegetables.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    image_channels = pupil.split_image_into_channels(image=image)

    # Convert ListOfImages → list[Image]
    channel_images = image_channels.to_list()

    # Convert to numpy
    channel_np_list = [img.to_numpy() for img in channel_images]

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_np_list),
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, channel_np_list)


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


def visualize(image: datatypes.Image, channel_np_list) -> None:
    """Visualizes the image and its channels using Rerun."""
    rr.init("split_image_into_channels", spawn=True)

    # Create Spatial2DViews dynamically
    views = [rrb.Spatial2DView(name="Original", origin="input")]
    # Order of channels for this example is (R, G, B)
    channel_names = ["Red", "Green", "Blue"]
    for i in range(len(channel_np_list)):
        views.append(
            rrb.Spatial2DView(
                name=channel_names[i],
                origin=f"channel_{i + 1}",
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

    # Log original image
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))

    # Log each channel
    for i, channel_np in enumerate(channel_np_list):
        rr.log(f"channel_{i + 1}", rr.Image(channel_np))


if __name__ == "__main__":
    split_image_into_channels_example()
