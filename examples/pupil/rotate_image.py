"""
Demonstrates rotate_image operation.

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


def rotate_image_example():
    """Applies rotate_image operation."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/synthetic_data_bin.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    transformed_image = pupil.rotate_image(
        image=image,
    angle_in_deg=10,
    interpolation_method="linear",
    keep_image_size=True,
    )

    transformed_image_np = transformed_image.to_numpy()
    logger.success(
        "Applied rotate_image. Output image shape: {}", 
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
    rr.init("rotate_image", spawn=True)
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
    rotate_image_example()
