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
import rerun.blueprint as rrb

from telekinesis import pupil


def project_pixel_to_camera_point_example():
    """Projects a pixel and depth to a 3D camera point."""
    # ===================== Create Parameters ==========================================
    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    pixel = np.array([320.0, 240.0], dtype=np.float64)
    depth = 1.0

    # ===================== Run Skill ==========================================
    camera_T_point = pupil.project_pixel_to_camera_point(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        pixel=pixel,
        depth=depth,
    )

    logger.success(
        "Projected pixel to camera point. camera_T_point shape: {}",
        np.asarray(camera_T_point.matrix).shape
        if hasattr(camera_T_point, "matrix")
        else "N/A",
    )

    # ===================== Visualization  (Optional) ======================
    visualize(camera_T_point)


def visualize(camera_T_point) -> None:
    """Visualizes the 3D camera point using Rerun."""
    rr.init("project_pixel_to_camera_point", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial3DView(name="Camera Point", origin="result")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    matrix = (
        camera_T_point.matrix
        if hasattr(camera_T_point, "matrix")
        else np.asarray(camera_T_point)
    )
    point_3d = matrix[:3, 3].reshape(1, 3).astype(np.float32)
    rr.log("result", rr.Points3D(positions=point_3d, colors=(0, 255, 0)))


if __name__ == "__main__":
    project_pixel_to_camera_point_example()
