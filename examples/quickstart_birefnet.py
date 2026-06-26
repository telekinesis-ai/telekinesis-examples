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
    "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/screws_standing.jpg"
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

    # Skill call: BiRefNet returns an annotation object we convert to a plain dict for logging
    result = cornea.segment_image_using_foreground_birefnet(
        image=image,
        mask_threshold=0,
    )
    annotation = result.to_dict()
    logger.success("BiRefNet foreground segmentation complete.")

    # Opens the Rerun viewer (spawn=True).
    rr.init("telekinesis_birefnet_quickstart", spawn=True)

    # Log input image and segmented image as separate entities.
    rr.log("input", rr.Image(image.to_numpy()))
    rr.log("segmented_mask", rr.Image(annotation["labeled_mask"]))


if __name__ == "__main__":
    main()