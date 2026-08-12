"""
Demonstrates bitwise NOT operation on an image.

This example:
- Downloads an example image.
- Performs bitwise NOT (inversion) operation.
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


def bitwise_not_image_example():
    """Performs bitwise NOT (inversion) on an image."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/einstein.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    result = pupil.bitwise_not_image(image=image)

    result_np = result.to_numpy()
    logger.success("Bitwise NOT. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================
    visualize(image, result_np)


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


def visualize(image: datatypes.Image, result_np) -> None:
    """Visualizes the original and inverted images using Rerun."""
    rr.init("bitwise_not_image", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Inverted", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("result", rr.Image(result_np))


if __name__ == "__main__":
    bitwise_not_image_example()
