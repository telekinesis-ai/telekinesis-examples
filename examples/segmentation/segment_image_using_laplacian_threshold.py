"""
Demonstrates Laplacian threshold segmentation.

This example:
- Downloads an example image.
- Uses the Laplacian operator to highlight high second-order intensity changes.
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

def segment_image_using_laplacian_threshold_example():
    """Uses the Laplacian operator to segment edge-rich areas."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/mechanical_parts_gray.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    annotations = cornea.segment_image_using_laplacian_threshold(image=image)
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
    rr.init("cornea_laplacian_threshold_segmentation", spawn=True)
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
    segment_image_using_laplacian_threshold_example()
