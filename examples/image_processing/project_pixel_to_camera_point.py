"""
Demonstrates projecting a pixel and depth to a 3D camera point.

This example:
- Creates camera intrinsics and distortion coefficients.
- Projects a pixel with depth to a 3D point in camera coordinates.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def project_pixel_to_camera_point_example():
    """Projects a pixel and depth to a 3D camera point."""
    # ===================== Create Parameters ==========================================
    # Camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], 
         [0, 500.0, 240.0], 
         [0, 0, 1.0]],
        dtype=np.float64,
    )

    # Distortion coefficients: [k1, k2, p1, p2, k3]
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    # Pixel: [u, v] in pixel coordinates
    pixel = np.array([320.0, 240.0], dtype=np.float64)

    # Depth: scalar value representing the distance from the camera to the point
    depth = 1.0

    # ===================== Run Skill ==========================================
    # Returns 4x4 transformation matrix representing the camera-to-point transformation
    camera_T_point = pupil.project_pixel_to_camera_point(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        pixel=pixel,
        depth=depth,
    )

    logger.success(
        "Projected pixel to camera point. camera_T_point: {}",
        camera_T_point.data,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("project_pixel_to_camera_point_example", spawn=True)
    datatypes.visualize(camera_T_point, entity_path="1-Camera Point")

if __name__ == "__main__":
    project_pixel_to_camera_point_example()
