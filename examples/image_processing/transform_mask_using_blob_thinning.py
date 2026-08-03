"""
Demonstrates blob thinning (skeletonization) transformation.

This example:
- Downloads an example image.
- Applies blob thinning operation.
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


def transform_mask_using_blob_thinning_example():
    """Applies blob thinning transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/handwriting_mask.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning guohall",
    )

    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied blob thinning. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, filtered_image_np)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads a binary image from a given URL and returns it as a telekinesis.datatypes.Image object.
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
    rr.init("transform_mask_using_blob_thinning", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Thinned", origin="filtered_image"),
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
    transform_mask_using_blob_thinning_example()
