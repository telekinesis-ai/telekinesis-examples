"""Demonstrates estimate_planar_poses operation."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def estimate_planar_poses_example():
    """Estimates 3D planar poses from a mask and depth image."""
    # ===================== Create Parameters ==========================================
    mask = datatypes.SegmentationImage.from_url(
        "https://assets.telekinesis.ai/examples/v1/images/permanent_marker_pen.png"
    )
    # Depth image saved as mm
    depth_image_mm = datatypes.DepthImage.from_url(
        "https://assets.telekinesis.ai/examples/v1/depth_images/permanent_marker_pen_depth.png"
    )
    depth_image_m = depth_image_mm.depth.astype(np.float32) / 1000.0

    # Camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_intrinsics = np.array(
        [
            [435.46856689, 0.0, 420.72525024],
            [0.0, 434.52377319, 244.55136108],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Distortion coefficients: [k1, k2, p1, p2, k3]
    distortion_parameters = np.array(
        [-0.05338378, 0.0578178, 0.00045024, 0.00018894, -0.018924], dtype=np.float64
    )
    camera_calibration = datatypes.CameraCalibration(
        height=mask.height,
        width=mask.width,
        distortion_model="plumb_bob",
        distortion_parameters=distortion_parameters,
        intrinsic_matrix=camera_intrinsics,
    )

    # ===================== Run Skill ==========================================
    camera_object_pose = pupil.estimate_planar_pose(
        mask=mask, depth_image=depth_image_m, camera_calibration=camera_calibration
    )

    # ===================== Log ================================================
    logger.success(f"Estimated planar poses using mask of shape {mask.shape}")
    logger.success(f"Result: {camera_object_pose}")

    # ===================== Visualization  (Optional) ======================
    rr.init("estimate_planar_poses_example", spawn=True)
    datatypes.visualize(mask, entity_path="/1-mask")
    datatypes.visualize(depth_image_mm, entity_path="/2-depthimage")
    datatypes.visualize(camera_object_pose, entity_path="/3-estimated3dpose")


if __name__ == "__main__":
    estimate_planar_poses_example()
