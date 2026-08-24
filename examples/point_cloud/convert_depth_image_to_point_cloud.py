"""
Demonstrates back-projecting a depth image into a 3D point cloud using a pinhole camera model.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def convert_depth_image_to_point_cloud_example():
    """
    Back-projects a depth image into a 3D point cloud using a pinhole camera model.

    For each pixel (u, v) holding a depth value Z, computes the 3D point
    X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.
    """
    # ===================== Load Data ==========================================
    depth_image = datatypes.DepthImage(
        np.full((480, 640), 1.5, dtype=np.float32)  # a flat wall 1.5m away
    )
    intrinsic_matrix = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # ===================== Run Skill ==========================================
    point_cloud = vitreous.convert_depth_image_to_point_cloud(
        depth_image=depth_image,
        intrinsic_matrix=intrinsic_matrix,
    )

    # ===================== Log ================================================
    logger.success(f"Converted {depth_image} to a point cloud")
    logger.success(f"Results: {point_cloud}")
    logger.info(f"Point cloud has {len(point_cloud.positions)} points")

    # ===================== Visualization  (Optional) ===========================
    rr.init("convert_depth_image_to_point_cloud_example", spawn=True)
    datatypes.visualize(depth_image, entity_path="/1-depth_image")
    datatypes.visualize(point_cloud, entity_path="/2-point_cloud")


if __name__ == "__main__":
    convert_depth_image_to_point_cloud_example()
