"""
Demonstrates cropping an image using multiple bounding boxes.

This example:
- Downloads an example image.
- Crops regions using multiple bounding boxes.
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


def crop_image_using_bounding_boxes_example():
    """Crops image using bounding boxes."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/driver_screw.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    bounding_boxes = [
        [65, 235, 330, 240],
        [370, 35, 330, 155],
        [445, 210, 85, 300],
    ]

    cropped_images = pupil.crop_image_using_bounding_boxes(
        image=image,
        bounding_boxes=bounding_boxes,
        retain_coordinates=True,
    )

    num_crops = len(cropped_images.to_list())
    logger.success("Cropped {} regions", num_crops)

    # ===================== Visualization  (Optional) ======================
    visualize(image, cropped_images)


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


def visualize(image: datatypes.Image, cropped_images) -> None:
    """Visualizes the original and cropped images using Rerun."""
    rr.init("crop_image_using_bounding_boxes", spawn=True)

    num_crops = len(cropped_images.to_list())
    views = [rrb.Spatial2DView(name="Original", origin="input")]

    for i in range(num_crops):
        views.append(
            rrb.Spatial2DView(name=f"Crop {i + 1}", origin=f"crops/{i}")
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(*views),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))

    # Log cropped images
    for i, crop in enumerate(cropped_images.to_list()):
        rr.log(f"crops/{i}", rr.Image(crop.to_numpy()))


if __name__ == "__main__":
    crop_image_using_bounding_boxes_example()
