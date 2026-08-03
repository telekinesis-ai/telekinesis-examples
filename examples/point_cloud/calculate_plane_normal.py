"""
Demonstrates extracting the normal vector from plane coefficients.

This example:
- Extracts and normalizes the normal vector from plane equation coefficients.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from telekinesis import vitreous


def calculate_plane_normal_example():
    """
    Extracts the normal vector from plane coefficients.

    Demonstrates extracting and normalizing the normal vector from plane equation
    coefficients (ax + by + cz + d = 0).
    """
    # ===================== Run Skill ==========================================
    normal_vector = vitreous.calculate_plane_normal(plane_coefficients=[0.0, 0.0, 1.0, 0.0])
    logger.success(
        f"Calculated normal vector to {normal_vector}"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(normal_vector)


def visualize(normal_vector) -> None:
    """Visualizes the plane and its normal vector using Rerun."""
    # Initialize Rerun
    rr.init("calculate_plane_normal", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup camera view
    overview_position = np.array([100.0, -100.0, 100.0])
    look_target = np.array([0.0, 0.0, 0.0])
    eye_up = np.array([0.0, 0.0, 1.0])

    # Add EyeControls3D with all parameters for camera movement tuning
    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,  # Camera control type: Orbital or FirstPerson
        position=overview_position,  # Initial camera position (None = auto)
        look_target=look_target,  # Point the camera looks at (None = auto)
        eye_up=eye_up,  # Up direction vector (None = auto)
        spin_speed=0.5,  # Speed of camera rotation/spin
        speed=0.0,  # Translation speed of camera movement
        tracking_entity=None,  # Entity to track (None = no tracking)
    )

    line_grid = rrb.LineGrid3D(
        visible=False,  # The grid is enabled by default, but you can hide it with this property.
    )

    spatial_information = rrb.SpatialInformation(
        target_frame="tf#/",
        show_axes=False,
        show_bounding_box=False,
    )
    background = rrb.Background(color=(255, 255, 255))  # White background

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(
            name="Plane with Normal Vector",
            origin="plane_visualization",
            background=background,
            eye_controls=eye_controls,
            line_grid=line_grid,
            spatial_information=spatial_information,
        )
    ))

    # Extract plane parameters: ax + by + cz + d = 0
    plane_point = np.array([0, 0, 0])

    # Create two orthogonal vectors in the plane
    if abs(normal_vector[2]) < 0.9:
        u = np.cross(normal_vector, np.array([0, 0, 1]))
    else:
        u = np.cross(normal_vector, np.array([1, 0, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vector, u)
    v = v / np.linalg.norm(v)

    # Create a grid of points on the plane to visualize it
    plane_size = 100.0
    grid_density = 20
    plane_points = []
    for i in np.linspace(-plane_size, plane_size, grid_density):
        for j in np.linspace(-plane_size, plane_size, grid_density):
            point = plane_point + i * u + j * v
            plane_points.append(point)

    # Log the plane as points
    rr.log("plane_visualization/plane", rr.Points3D(
        np.array(plane_points),
        colors=[[200, 200, 200]] * len(plane_points),
        radii=[0.5] * len(plane_points)
    ))

    # Log the plane point (origin on the plane)
    rr.log("plane_visualization/plane_point", rr.Points3D(
        np.array([plane_point]),
        colors=[[255, 255, 0]],
        radii=[2.0]
    ))

    # Log the normal vector as an arrow from the plane point
    normal_length = 50.0
    rr.log("plane_visualization/normal_vector", rr.Arrows3D(
        origins=np.array([plane_point]),
        vectors=np.array([normal_vector * normal_length]),
        colors=np.array([[255, 0, 0]])
    ))


if __name__ == "__main__":
    calculate_plane_normal_example()
