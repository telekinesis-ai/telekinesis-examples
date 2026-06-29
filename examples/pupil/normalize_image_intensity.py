"""
Demonstrates normalize_image_intensity operation.

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


def normalize_image_intensity_example():
    """Applies normalize_image_intensity operation."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/gauge_washed.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.normalize_image_intensity(
        image=image,
    alpha=0.0,
    beta=255.0,
    normalization_method="minmax",
    output_format="8bit",
    )

    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied normalize_image_intensity. Output image shape: {}", 
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, filtered_image_np)


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


def visualize(image: datatypes.Image, filtered_image_np) -> None:
    """Visualizes the original and filtered images using Rerun."""
    rr.init("normalize_image_intensity", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Converted", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


if __name__ == "__main__":
    normalize_image_intensity_example()
