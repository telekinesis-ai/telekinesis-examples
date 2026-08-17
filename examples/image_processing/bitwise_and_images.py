"""Demonstrates bitwise AND operation between two images."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def bitwise_and_images_example():
    """Performs bitwise AND between two images."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/bin_picking_metal_2.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Create Mask ==========================================
    bbox = [450, 210, 1040, 616]
    x1, y1, x2, y2 = bbox
    mask_data = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_data[y1:y2, x1:x2] = 255
    mask = datatypes.Image(mask_data)

    # ===================== Run Skill ==========================================
    result_image = pupil.bitwise_and_images(image_a=image, image_b=mask)

    # ===================== Log ================================================
    logger.success(f"Bitwise AND between {image} and {mask} completed.")
    logger.success(f"Result: {result_image}")

    # ===================== Visualization  (Optional) ==========================
    rr.init("bitwise_and_images_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(mask, entity_path="2-Mask")
    datatypes.visualize(result_image, entity_path="3-Filtered")

if __name__ == "__main__":
    bitwise_and_images_example()
