"""Demonstrates projecting a 3D world point to pixel coordinates."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def project_world_point_to_pixel_example():
    """Projects a 3D world point to pixel coordinates."""
    # ===================== Create Parameters ==========================================
    # Point: [x, y, z] in world coordinates
    point = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Camera-to-world transformation matrix (4x4)
    world_T_camera = np.eye(4, dtype=np.float64)
    world_T_camera[2, 3] = 1.0

    # Camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    # Distortion coefficients: [k1, k2, p1, p2, k3]
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    # ===================== Run Skill ==========================================
    pixel = pupil.project_world_point_to_pixel(
        point=point,
        world_T_camera=world_T_camera,
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
    )

    # ===================== Log ================================================
    logger.success(f"Projected world point to pixel using {point}")
    logger.success(f"Result: {pixel}")

    # ===================== Visualization  (Optional) ======================
    rr.init("project_world_point_to_pixel_example", spawn=True)
    datatypes.visualize(pixel, entity_path="1-pixel")

if __name__ == "__main__":
    project_world_point_to_pixel_example()
