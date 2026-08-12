"""
Demonstrates centroid calculation on a binary mask.

This example:
- Downloads an example binary image.
- Computes the centroid of non-zero pixels.
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


def calculate_image_centroid_example():
    """Computes the centroid of a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/metal_part_mask.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    centroid = pupil.calculate_image_centroid(mask=image)

    centroid_pos = centroid.to_numpy().reshape(-1, 2)
    logger.success(
        "Computed centroid. Position: ({}, {})",
        centroid_pos[0, 0],
        centroid_pos[0, 1],
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, centroid_pos)


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


def visualize(image: datatypes.Image, centroid_pos) -> None:
    """Visualizes the binary mask with centroid using Rerun."""
    rr.init("calculate_image_centroid", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Mask", origin="mask"),
                rrb.Spatial2DView(name="Centroid", origin="centroid"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    mask_np = image.to_numpy()
    rr.log("mask", rr.Image(mask_np))
    rr.log("centroid", rr.Image(mask_np))
    rr.log(
        "centroid/point",
        rr.Points2D(positions=centroid_pos, radii=4, colors=[[0, 255, 0]]),
    )


if __name__ == "__main__":
    calculate_image_centroid_example()
