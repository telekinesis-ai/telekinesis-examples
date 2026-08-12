"""
Demonstrates overlaying COCOObjectDetectionAnnotations on an Image.

This example:
- Builds an RGB Image.
- Builds COCOObjectDetectionAnnotations sized to that image.
- Logs both under the same entity path via `visualize(image, detections, ...)`
  so the boxes and masks render on top of the image in a shared Rerun view.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def object_detection_overlay_example():
    # An image the detections are defined against.
    H, W = 720, 1280
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:] = (30, 30, 40)  # dark backdrop so overlays are visible
    my_image = datatypes.Image(image_array)

    # Detections in this image's pixel space: [x, y, w, h].
    my_detections = datatypes.COCOObjectDetectionAnnotations(
        ids=np.array([0, 1], dtype=np.int32),
        image_ids=np.array([0, 0], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        bboxes=np.array([[100, 100, 300, 200], [700, 400, 250, 250]], dtype=np.float32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
    )
    logger.info(f"Overlaying {len(my_detections.bboxes)} detections on a {W}x{H} image")

    # Both log under the same entity path; the detections overlay the image.
    logger.info("Visualizing with Rerun...")
    rr.init("object_detection_overlay", spawn=True)
    datatypes.visualize(my_image, my_detections, entity_path="/image")


if __name__ == "__main__":
    object_detection_overlay_example()
