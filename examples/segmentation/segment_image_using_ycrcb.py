"""
Demonstrates YCrCb color space segmentation.

This example:
- Downloads an example image.
- Segments it using YCrCb color space range.
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


def segment_image_using_ycrcb_example():
    """Segments an image using YCrCb color space range."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/David_Schwimmer.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    annotations = cornea.segment_image_using_ycrcb(
        image=image, lower_bound=(0, 133, 77), upper_bound=(255, 173, 127)
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
    rr.init("cornea_ycrcb_segmentation", spawn=True)
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
    segment_image_using_ycrcb_example()
