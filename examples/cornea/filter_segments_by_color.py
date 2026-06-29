"""
Demonstrates filtering superpixels based on color.

This example:
- Downloads an example image.
- Segments it using Felzenszwalb algorithm.
- Filters segments based on color criteria.
- Visualizes the filtered segmentation using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb
from datatypes import datatypes
from telekinesis import cornea, pupil

def filter_segments_by_color_example():
    """Filters superpixels based on color criteria."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/eggs_carton.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    annotations = cornea.segment_image_using_felzenszwalb(
        image=image, scale=500, sigma=1, min_size=200
    )
    annotations_dict = annotations.to_dict()
    superpixel_labels = annotations_dict["labeled_mask"]
    superpixel_labels = datatypes.Image(superpixel_labels)
    annotations = cornea.filter_segments_by_color(
        image=image, labels=superpixel_labels, min_color=0, max_color=125.0
    )
    annotations_dict = annotations.to_dict()
    logger.success("Filtering completed.")

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
    """Visualizes the original image and the filtered segmentation mask using Rerun."""
    rr.init("cornea_filter_segmentation_by_color", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    filtered_labels_np = annotations_dict["labeled_mask"]
    rr.log("input", rr.Image(image_np))
    rr.log("segmented_mask", rr.SegmentationImage(filtered_labels_np))

if __name__ == "__main__":
    filter_segments_by_color_example()
