"""Demonstrates estimate_planar_poses operation."""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def estimate_planar_poses_example():
    """Estimates 3D planar poses from a mask and depth image."""
    # ===================== Create Parameters ==========================================
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    mask = datatypes.SegmentationImage.from_url(
        "https://assets.telekinesis.ai/examples/v1/images/xyzibd_flat_bar_pose_mask.png"
    )
    # Depth image saved as mm
    depth_image_mm = datatypes.DepthImage.from_url(
        "https://assets.telekinesis.ai/examples/v1/depth_images/xyzibd_flat_bar_pose_depth.png"
    )
    depth_image_m = depth_image_mm.depth.astype(np.float32) / 1000.0

    # Camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_intrinsics = np.array(
        [
            [2329.03738260, 0.0, 1056.72436003],
            [0.0, 2329.79188963, 721.69460828],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Distortion coefficients: [k1, k2, p1, p2, k3]
    distortion_parameters = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
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
