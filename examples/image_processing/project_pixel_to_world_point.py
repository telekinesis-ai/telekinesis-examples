"""Demonstrates projecting a pixel and depth to a 3D world point."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def project_pixel_to_world_point_example():
    """Projects a pixel and depth to a 3D world point."""
    # ===================== Create Parameters ==========================================
    # Pixel: [u, v] in pixel coordinates
    pixel = np.array([320.0, 240.0], dtype=np.float64)

    # Depth: scalar value representing the distance from the camera to the point
    depth = 1.0

    # Camera-to-world transformation matrix (4x4)
    world_T_camera = np.eye(4, dtype=np.float64)
    world_T_camera[2, 3] = 1.0
    
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

    # ===================== Run Skill ==========================================
    world_point = pupil.project_pixel_to_world_point(
        pixel=pixel,
        depth=depth,
        world_T_camera=world_T_camera,
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
    )

    # ===================== Log ================================================
    logger.success(f"Projected pixel to world point using {pixel} and depth {depth}")
    logger.success(f"Result: {world_point}")

    # ===================== Visualization  (Optional) ======================
    rr.init("project_pixel_to_world_point_example", spawn=True)
    datatypes.visualize(world_point, entity_path="1-worldpoint")

if __name__ == "__main__":
    project_pixel_to_world_point_example()
