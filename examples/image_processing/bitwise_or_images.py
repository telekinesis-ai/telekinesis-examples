"""
Demonstrates bitwise OR operation between two images.

This example:
- Downloads two example images.
- Performs bitwise OR operation.
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


def bitwise_or_images_example():
    """Performs bitwise OR between two images."""
    # ===================== Load Images ==========================================
    image_url_a = "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    image_url_b = "https://assets.telekinesis.ai/examples/v1/images/rectangles_mask.png"
    image_a = fetch_image(image_url_a)
    image_b = fetch_image(image_url_b)

    # ===================== Resize Image B ==========================================
    image_b = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
        pad_color=(0, 0, 0),
    )

    # ===================== Run Skill ==========================================
    result = pupil.bitwise_or_images(image_a=image_a, image_b=image_b)

    result_np = result.to_numpy()
    logger.success("Bitwise OR. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================
    visualize(image_a, image_b, result_np)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads a binary image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_bgr, color_model="L")
    logger.success(f"Loaded image from {image_url}")
    return image


def visualize(image_a: datatypes.Image, image_b: datatypes.Image, result_np) -> None:
    """Visualizes the input images and result using Rerun."""
    rr.init("bitwise_or_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise OR", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = image_b.to_numpy()
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result_np))


if __name__ == "__main__":
    bitwise_or_images_example()
