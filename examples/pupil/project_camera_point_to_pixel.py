"""
Demonstrates projecting a 3D camera point to pixel coordinates.

This example:
- Creates camera intrinsics and distortion coefficients.
- Projects a 3D point in camera coordinates to pixel coordinates.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import pupil


def project_camera_point_to_pixel_example():
    """Projects a 3D camera point to pixel coordinates."""
    # ===================== Create Parameters ==========================================
    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    point = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ===================== Run Skill ==========================================
    pixel = pupil.project_camera_point_to_pixel(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        point=point,
    )

    positions = pixel.to_numpy().reshape(-1, 2)
    logger.success("Projected camera point to pixel. Pixel: {}", positions)

    # ===================== Visualization  (Optional) ======================
    visualize(positions)


def visualize(positions) -> None:
    """Visualizes the projected pixel using Rerun."""
    rr.init("project_camera_point_to_pixel", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial2DView(name="Pixel", origin="pixel")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    canvas = np.ones((480, 640, 3), dtype=np.uint8)
    rr.log("pixel", rr.Image(canvas))
    rr.log(
        "pixel/projected_point",
        rr.Points2D(
            positions=positions,
            radii=6,
        ),
    )


if __name__ == "__main__":
    project_camera_point_to_pixel_example()
