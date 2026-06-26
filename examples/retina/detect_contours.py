"""
Detect contours using a contour-based detector.

Extracts contours from the input image and returns
coco-style annotations.

The annotations are used for visualization overlays.
"""

import numpy as np
import requests
import cv2

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import retina
from datatypes import datatypes


def detect_contours_example():
    """
    Detect contours using a contour-based detector.

    Extracts contours from the input image and returns
    coco-style annotations.

    The annotations are used for visualization overlays.
    """

    # ===================== Load Image ==========================================
    # Download and decode image from cloud
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/nuts_scattered_filtered_gaussian.png"
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_binary = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_binary, color_model="L")
    logger.success(f"Loaded image from {image_url}")

    # ===================== Run Skill ==========================================
    # Detect circles
    annotations = retina.detect_contours(
        image=image,
        retrieval_mode="retrieve_list",
        approx_method="chain_approximate_simple",
        min_area=200,
        max_area=100000,
    )

    # Access results
    annotations = annotations.to_list()
    logger.success(
        f"Detected {len(annotations)} contours using contour detector."
    )

    # ===================== Visualization  (Optional) ======================

    image_np = image.to_numpy()

    # Extract contours and bboxes form annotations
    contour_polylines = []
    bboxes = []
    for annotation in annotations:
        contour_dict = annotation["geometry"]
        points = contour_dict["points"]
        if not points:
            continue
        contour_polylines.append(np.array(points, dtype=np.float32))
        bboxes.append(annotation["bbox"])

    # Intialize Rerun and send blueprint
    rr.init("detect_contours_example", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log countours as LineStrips2D on overlay image
    contour_labels = [f"Contour {i}" for i in range(len(contour_polylines))]
    rr.log(
        "detection/contours",
        rr.LineStrips2D(
            contour_polylines,
            colors=[[0, 255, 0]],
            radii=[2],
            labels=contour_labels,
        ),
    )

    # Log bounding boxes using Boxes2D on overlay image
    box_labels = [f"Box {i}" for i in range(len(bboxes))]
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=bboxes,
            array_format=rr.Box2DFormat.XYWH,
            colors=[[0, 255, 0]],
            labels=box_labels,
            radii=[2],
        ),
    )


if __name__ == "__main__":
    detect_contours_example()
