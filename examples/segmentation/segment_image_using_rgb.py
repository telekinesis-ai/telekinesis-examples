"""
Demonstrates RGB color space segmentation.

This example:
- Downloads an example image.
- Segments it using RGB color space range.
- Visualizes the result using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from datatypes import datatypes
from telekinesis import cornea, pupil


def segment_image_using_rgb_example():
    """Segments an image using RGB color space range."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/cylinder_on_conveyor.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    annotations = cornea.segment_image_using_rgb(
        image=image, lower_bound=(0, 50, 50), upper_bound=(180, 255, 255)
    )
    annotations_dict = annotations.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================
    visualize(image, annotations_dict)


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

def visualize(image: datatypes.Image, annotations_dict: dict) -> None:
    """Visualizes the original image and the segmentation mask using Rerun."""
    rr.init("cornea_rgb_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    mask_np = annotations_dict["labeled_mask"]
    rr.log("input", rr.Image(image_np))
    rr.log("segmented_mask", rr.Image(mask_np))


if __name__ == "__main__":
    segment_image_using_rgb_example()
