"""
Demonstrates crop_image_center operation.

This example:
- Downloads an example image.
- Applies the operation.
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


def crop_image_center_example():
    """Applies crop_image_center operation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/rusted_metal_gear.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    transformed_image = pupil.crop_image_center(
        image=image,
    crop_width=300,
    crop_height=300,
    pad_color=(0, 0, 0),
    )

    transformed_image_np = transformed_image.to_numpy()
    logger.success(
        "Applied crop_image_center. Output image shape: {}", 
        transformed_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, transformed_image_np)


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


def visualize(image: datatypes.Image, transformed_image_np) -> None:
    """Visualizes the original and transformed images using Rerun."""
    rr.init("crop_image_center", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Transformed", origin="transformed_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("transformed_image", rr.Image(transformed_image_np))


if __name__ == "__main__":
    crop_image_center_example()
