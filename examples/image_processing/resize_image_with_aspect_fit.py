"""
Demonstrates resize_image_with_aspect_fit operation.

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


def resize_image_with_aspect_fit_example():
    """Applies resize_image_with_aspect_fit operation."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/gearbox.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    transformed_image = pupil.resize_image_with_aspect_fit(
        image=image,
    resize_width=400,
    resize_height=300,
    interpolation_method="linear",
    )

    transformed_image_np = transformed_image.to_numpy()
    logger.success(
        "Applied resize_image_with_aspect_fit. Output image shape: {}", 
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
    rr.init("resize_image_with_aspect_fit", spawn=True)
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
    resize_image_with_aspect_fit_example()
