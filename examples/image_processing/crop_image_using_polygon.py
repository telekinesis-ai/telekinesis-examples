"""
Demonstrates cropping an image using a polygon mask.

This example:
- Downloads an example image.
- Crops a polygonal region from the image.
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


def crop_image_using_polygon_example():
    """Crops image using a polygon mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    polygon_vertices = [
        [37, 404],
        [46, 373],
        [74, 323],
        [106, 258],
        [125, 154],
        [165, 106],
        [200, 115],
        [210, 173],
        [206, 199],
        [250, 208],
        [193, 255],
        [216, 331],
        [240, 383],
        [250, 411],
    ]

    cropped_image = pupil.crop_image_using_polygon(
        image=image,
        polygon_vertices=polygon_vertices,
    )

    cropped_image_np = cropped_image.to_numpy()
    logger.success(
        "Cropped image (polygon). Output shape: {}",
        cropped_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, cropped_image_np)


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


def visualize(image: datatypes.Image, cropped_image_np) -> None:
    """Visualizes the original and cropped images using Rerun."""
    rr.init("crop_image_using_polygon", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Cropped", origin="cropped_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("cropped_image", rr.Image(cropped_image_np))


if __name__ == "__main__":
    crop_image_using_polygon_example()
