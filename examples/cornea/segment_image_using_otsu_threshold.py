"""
Demonstrates Otsu threshold segmentation.

This example:
- Downloads an example image.
- Applies Otsu's method to find a global threshold.
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

def segment_image_using_otsu_threshold_example():
    """Applies Otsu's method to find a global threshold for the image."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/buttons_arranged.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    annotations = cornea.segment_image_using_otsu_threshold(image=image)
    annotations_dict = annotations.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================
    visualize(image, annotations_dict)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads a grayscale image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_grayscale = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_grayscale, color_model="L")
    logger.success(f"Loaded image from {image_url}")
    return image

def visualize(image: datatypes.Image, annotations_dict: dict) -> None:
    """Visualizes the original image and the segmentation mask using Rerun."""
    rr.init("cornea_otsu_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Input", origin="input"),
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
    rr.log("segmented_mask", rr.SegmentationImage(mask_np))

if __name__ == "__main__":
    segment_image_using_otsu_threshold_example()
