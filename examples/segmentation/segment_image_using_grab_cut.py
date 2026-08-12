"""
Demonstrates GrabCut segmentation.

This example:
- Downloads an example image.
- Segments it using the GrabCut algorithm with a bounding box.
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

def segment_image_using_grab_cut_example():
    """Segments an image using the GrabCut algorithm."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/plastic_part.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    bbox = [220, 20, 930, 850]
    bbox_dt = datatypes.Boxes2D(arrays=bbox, array_format="XYWH")
    annotations = cornea.segment_image_using_grab_cut(
        image=image, num_iterations=2, bbox=bbox_dt
    )
    annotations_dict = annotations.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================
    visualize(image, bbox, annotations_dict)


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

def visualize(image: datatypes.Image, bbox: list, annotations_dict: dict) -> None:
    """Visualizes the original image with bounding box and the segmentation mask using Rerun."""
    rr.init("cornea_grab_cut_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image", origin="input"),
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
    rr.log("input", rr.Boxes2D(array=np.array([bbox]), array_format=rr.Box2DFormat.XYWH, colors=[0, 255, 0]))
    rr.log("segmented_mask", rr.Image(mask_np))

if __name__ == "__main__":
    segment_image_using_grab_cut_example()
