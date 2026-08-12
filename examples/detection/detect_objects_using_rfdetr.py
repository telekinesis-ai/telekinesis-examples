"""
Detect objects using RF-DETR.

Runs RF-DETR object detection on an image and returns COCO-like annotations
with category names from the COCO 80-class label set.

The annotations and categories are used for visualization overlays.
"""

import numpy as np
import requests
import cv2

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import retina, pupil
from datatypes import datatypes


def detect_objects_using_rfdetr_example():
    """
    Detect objects using RF-DETR.

    Runs RF-DETR object detection on an image and returns COCO-like annotations
    with category names from the COCO 80-class label set.

    The annotations and categories are used for visualization overlays.
    """

    # ===================== Load Image ==========================================

    # Download and decode image from cloud
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/warehouse_1.jpg"
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    image = datatypes.Image(image=image_bgr, color_model="BGR")
    image = pupil.convert_image_color_space(image, 
                                            source_color_space="BGR", 
                                            target_color_space="RGB") 
    logger.success(f"Loaded image from {image_url}")

    # ===================== Run Skill ==========================================

    annotations, categories = retina.detect_objects_using_rfdetr(
        image=image,
        score_threshold=0.5,
    )

    # Access results
    annotations = annotations.to_list()
    categories = categories.to_list()
    logger.success(f"RF-DETR detected {len(annotations)} objects.")

    # ===================== Visualization  (Optional) ======================

    image_np = image.to_numpy()

    # Build categories_map
    categories_map = {
        category["id"]: category["name"] for category in categories
    }

    # Extract objects form annotations
    bboxes = []
    colors = []
    labels = []
    radii = []
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, ann in enumerate(annotations):
        color = colors_list[idx % len(colors_list)]
        label = categories_map.get(ann.get("category_id", 0), "")
        score = ann.get("score", 0.0)
        bboxes.append(ann["bbox"])  # [x, y, w, h]
        colors.append(color)  # (r,g,b)
        labels.append(f"{label}{score:.2f}")
        radii.append(2)

    # Intialize Rerun and send blueprint
    rr.init("detect_objects_using_rfdetr_example", spawn=True)

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

    # Log bounding boxes as Boxes2D on overlay image
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=np.array(bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            colors=np.array(colors, dtype=np.uint8),
            labels=labels,
            radii=radii,
        ),
    )


if __name__ == "__main__":
    detect_objects_using_rfdetr_example()
