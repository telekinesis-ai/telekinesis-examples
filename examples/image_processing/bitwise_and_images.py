"""
Demonstrates bitwise AND operation between two images.

This example:
- Downloads an example image.
- Creates a mask from the image.
- Performs bitwise AND operation.
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


def bitwise_and_images_example():
    """Performs bitwise AND between two images."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/bin_picking_metal_2.jpg"
    image_a = fetch_image(image_url)

    # ===================== Create Mask ==========================================
    image_a_np = image_a.to_numpy()
    bbox = [450, 210, 1040, 616]
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image_a_np.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # ===================== Run Skill ==========================================
    result = pupil.bitwise_and_images(image_a=image_a, image_b=mask)

    result_np = result.to_numpy()
    logger.success("Bitwise AND. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================
    visualize(image_a, mask, result_np)


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


def visualize(image_a: datatypes.Image, image_b, result_np) -> None:
    """Visualizes the input images and result using Rerun."""
    rr.init("bitwise_and_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise AND", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = image_b
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result_np))


if __name__ == "__main__":
    bitwise_and_images_example()
