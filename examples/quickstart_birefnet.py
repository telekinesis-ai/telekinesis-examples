"""Telekinesis quickstart: extract the foreground from an image with BiRefNet.

Loads a sample image from a public URL, runs BiRefNet foreground
segmentation, and visualizes the input alongside the predicted mask
in a Rerun viewer.

Run as a script - python quickstart_birefnet_example.py
"""

import cv2
import numpy as np
import requests
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger

from datatypes import datatypes
from telekinesis import cornea, pupil


# Public sample image shipped by Telekinesis (you can swap this for a local file later).
IMAGE_URL = (
    "https://assets.telekinesis.ai/screws_standing.jpg"
)


def main() -> None:
    # Download and decode JPEG bytes to BGR, then convert to RGB for datatypes.Image
    response = requests.get(IMAGE_URL, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR,
    )

    image_bgr = datatypes.Image(image=image_bgr, color_model="BGR")
    image = pupil.convert_image_color_space(image_bgr, source_color_space="BGR", target_color_space="RGB")
    # Wrap pixels in Telekinesis's typed Image object.
    logger.success(f"Loaded image from {IMAGE_URL}")

    # Skill call: BiRefNet returns a datatypes.SegmentationImage label map (0 = background, 1 = foreground)
    segmented_image = cornea.segment_image_foreground_using_birefnet(
        image=image,
        mask_threshold=0,
    )
    logger.success("BiRefNet foreground segmentation complete.")

    # Opens the Rerun viewer (spawn=True).
    rr.init("telekinesis_birefnet_quickstart", spawn=True)

    # Log input image and segmented image as separate entities.
    rr.log("input", rr.Image(image.to_numpy()))
    rr.log("segmented_mask", rr.Image(segmented_image.data))


if __name__ == "__main__":
    main()