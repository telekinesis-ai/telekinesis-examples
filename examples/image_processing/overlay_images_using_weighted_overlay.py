"""
Demonstrates weighted overlay blending of two images.

This example:
- Downloads an example image.
- Creates a rotated version for blending.
- Performs weighted overlay operation.
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


def overlay_images_using_weighted_overlay_example():
    """Blends two images using weighted overlay."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/rusted_metal_gear.jpg"
    image_a = fetch_image(image_url)
    image_a = pupil.resize_image_with_aspect_fit(
        image=image_a,
        resize_width=512,
        resize_height=512,
    )

    # ===================== Create Second Image ==========================================
    image_b = pupil.rotate_image(
        image=image_a, angle_in_deg=60.0, keep_image_size=True
    )
    logger.success(f"Loaded image from {image_url}")

    # ===================== Run Skill ==========================================
    blended = pupil.overlay_images_using_weighted_overlay(
        image_a=image_a,
        image_b=image_b,
        weight_a=0.5,
        weight_b=0.5,
    )

    blended_np = blended.to_numpy()
    logger.success("Weighted overlay. Output shape: {}", blended_np.shape)

    # ===================== Visualization  (Optional) ======================
    visualize(image_a, image_b, blended_np)


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


def visualize(image_a: datatypes.Image, image_b: datatypes.Image, blended_np) -> None:
    """Visualizes the input images and blended result using Rerun."""
    rr.init("overlay_images_using_weighted_overlay", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Blended", origin="blended"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image_a.to_numpy()
    image_2_np = image_b.to_numpy()
    rr.log("input_a", rr.Image(image_np))
    rr.log("input_b", rr.Image(image_2_np))
    rr.log("blended", rr.Image(blended_np))


if __name__ == "__main__":
    overlay_images_using_weighted_overlay_example()
