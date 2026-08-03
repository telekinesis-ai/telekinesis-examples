"""
Detect circles using the classic Hough Circle Transform.

Runs Hough circle detection on a grayscale image and returns
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


def detect_circle_using_classic_hough_example():
    """
    Detect circles using the classic Hough Circle Transform.

    Runs Hough circle detection on a grayscale image and returns
    coco-style annotations.

    The annotations are used for visualization overlays.
    """
    # ===================== Load Image ==========================================
    # Download and decode image from cloud
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/metal_gears.jpg"
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_grayscale = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_grayscale, color_model="L")
    logger.success(f"Loaded image from {image_url}")

    # ===================== Run Skill ==========================================
    # Detect circles
    annotations = retina.detect_circle_using_classic_hough(
        image=image,
        inverse_resolution_ratio=1,
        min_distance=50,
        min_radius=40,
        max_radius=60,
        canny_detector_upper_threshold=300,
        accumulator_threshold=30,
    )

    # Access results
    annotations = annotations.to_list()
    logger.success(
        f"Detected {len(annotations)} circles using classic Hough transform."
    )

    # ===================== Visualization  (Optional) ===========================

    image_np = image.to_numpy()

    # Extract circles and bboxes form annotations
    circles = []
    bboxes = []
    for annotation in annotations:
        bboxes.append(annotation["bbox"])  # [x, y, w, h]
        circle_dict = annotation["geometry"]
        cx, cy = circle_dict["center"]
        r = circle_dict["radius"]
        circles.append((float(cx), float(cy), float(r)))

    # Intialize Rerun and send blueprint
    rr.init("classic_hough_circle_detector_example", spawn=True)
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

    # Build circle polylines using LineStrips2D
    def circle_polyline_2d(center_xy, radius, n=128):
        cx, cy = center_xy
        t = np.linspace(0, 2 * np.pi, n, endpoint=True)
        pts = np.stack(
            [cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1
        )
        return pts

    circle_polylines = [
        circle_polyline_2d((cx, cy), r) for cx, cy, r in circles
    ]
    circle_labels = [
        f"Circle {i} (r={int(r)})" for i, (cx, cy, r) in enumerate(circles)
    ]

    # Log circle outlines as LineStrips2D on overlay image
    rr.log(
        "detection/circles",
        rr.LineStrips2D(
            circle_polylines,
            colors=[[0, 255, 0]],
            radii=[1],
            labels=circle_labels,
        ),
    )

    # Log bounding boxes as Boxes2D on overlay image
    box_labels = [f"Box {i}" for i in range(len(bboxes))]
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=bboxes,
            array_format=rr.Box2DFormat.XYWH,
            colors=[[0, 255, 0]],
            labels=box_labels,
            radii=[1],
        ),
    )


if __name__ == "__main__":
    detect_circle_using_classic_hough_example()
