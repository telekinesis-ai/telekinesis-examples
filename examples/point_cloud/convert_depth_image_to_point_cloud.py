"""
Demonstrates converting a depth image to a point cloud using camera intrinsics.

This example:
- Downloads an example depth image.
- Projects the image into a 3D point cloud using camera intrinsics.
- Visualizes the depth image and point cloud using Rerun.
"""

import cv2
import numpy as np
import requests
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def convert_depth_image_to_point_cloud_example():
    """Converts a metric depth image to a point cloud using camera intrinsics."""
    # ===================== Load Data ==========================================
    depth_image_url = "https://assets.telekinesis.ai/examples/v1/depth_images/bin_picking.png"
    depth_image = fetch_depth_image(depth_image_url)
    logger.success(f"Loaded depth image from {depth_image_url}")

    # ===================== Run Skill ==========================================
    # Define the camera intrinsic matrix
    intrinsic_matrix = np.array(
        [
            [643.9369506835938, 0.0, 644.6060791015625],
            [0.0, 643.9369506835938, 354.50860595703125],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    point_cloud = vitreous.convert_depth_image_to_point_cloud(
        depth_image=depth_image,
        intrinsic_matrix=intrinsic_matrix,
    )
    logger.success(f"Converted depth image to {len(point_cloud)} points")

    # ===================== Visualization (Optional) ===========================
    rr.init("convert_depth_image_to_point_cloud_example", spawn=True)
    datatypes.visualize(depth_image, entity_path="/input_depth_image")
    datatypes.visualize(point_cloud, entity_path="/output_point_cloud")


def fetch_depth_image(url: str) -> datatypes.Image:
    """Downloads a depth image from a URL and loads it as an `Image` object.

    `Image.from_url` always decodes to RGB/RGBA `uint8`, which would clip a
    16-bit depth PNG, so the raw buffer is decoded with OpenCV
    (preserving full precision) and cast to `float32` instead.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    depth_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    return datatypes.Image(data=depth_array.astype(np.float32))


if __name__ == "__main__":
    convert_depth_image_to_point_cloud_example()
