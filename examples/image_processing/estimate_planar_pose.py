"""Demonstrates estimate_planar_poses operation."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def estimate_planar_poses_example():
    """Estimates 3D planar poses from a mask and depth image."""
    # ===================== Create Parameters ==========================================
    image = datatypes.Image.from_url(
        "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    )
    mask = cornea.segment_image_using_threshold(image=image)
    depth_image = datatypes.DepthImage(
        np.full(mask.shape, 1.5, dtype=np.float32)
    )

    # Camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64
    )
    # Distortion coefficients: [k1, k2, p1, p2, k3]
    distortion_parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    camera_calibration = datatypes.CameraCalibration(
        height=image.height,
        width=image.width,
        distortion_model="plumb_bob",
        distortion_parameters=distortion_parameters,
        intrinsic_matrix=camera_intrinsics,
    )

    # ===================== Run Skill ==========================================
    pose_3d = pupil.estimate_planar_pose(
        mask=mask,
        depth_image=depth_image,
        camera_calibration=camera_calibration
    )

    # ===================== Log ================================================
    logger.success(f"Estimated planar poses using mask of shape {mask.shape}")
    logger.success(f"Result: {pose_3d}")

    # ===================== Visualization  (Optional) ======================
    rr.init("estimate_planar_poses_example", spawn=True)
    datatypes.visualize(image, entity_path="0-inputimage")
    datatypes.visualize(mask, entity_path="1-mask")
    datatypes.visualize(depth_image, entity_path="2-depthimage")
    datatypes.visualize(pose_3d, entity_path="3-estimated3dpose")

if __name__ == "__main__":
    estimate_planar_poses_example()
