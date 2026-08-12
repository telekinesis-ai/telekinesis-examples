"""
Demonstrates filter_image_using_sobel operation.

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


def filter_image_using_sobel_example():
    """Applies filter_image_using_sobel operation."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/nuts.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.filter_image_using_sobel(
        image=image,
    output_format="64bit",
    dx=1,
    dy=1,
    kernel_size=9,
    scale=1.0,
    delta=0.0,
    border_type="default",
    )

    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied filter_image_using_sobel. Output image shape: {}", 
        filtered_image_np.shape,
    )
    # Handle float images for visualization
    filtered_image_np = np.abs(filtered_image_np.astype(np.float32))
    filtered_image_np = np.clip(filtered_image_np, 0, 255)
    filtered_image_np = filtered_image_np.astype(np.uint8)

    # ===================== Visualization  (Optional) ======================
    visualize(image, filtered_image_np)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads an image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_bgr, color_model="L")
    logger.success(f"Loaded image from {image_url}")
    return image


def visualize(image: datatypes.Image, filtered_image_np) -> None:
    """Visualizes the original and filtered images using Rerun."""
    rr.init("filter_image_using_sobel", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
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
    filter_image_using_sobel_example()
