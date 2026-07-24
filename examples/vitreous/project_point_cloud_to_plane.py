"""
Demonstrates projecting a point cloud orthogonally onto a plane.

This example:
- Downloads an example point cloud.
- Projects all points orthogonally onto a plane, flattening the cloud onto a 2D surface in 3D space.
- Visualizes the result using Rerun.
"""

import pathlib
import tempfile

import numpy as np
import requests
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from datatypes import datatypes, io
from telekinesis import vitreous


def project_point_cloud_to_plane_example():
    """
    Projects all points orthogonally onto a plane.

    Moves each point to its closest point on the specified plane. Flattens
    the cloud onto a 2D surface in 3D space.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    projected_point_cloud = vitreous.project_point_cloud_to_plane(
        add_white_noise=False,
        white_noise_standard_deviation=1e-6,
        point_cloud=point_cloud,
        plane_coefficients=[0.0, 0.0, 1.0, 0.0],
    )
    logger.success(f"Projected {len(projected_point_cloud.positions)} points to plane")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, projected_point_cloud)


def fetch_point_cloud(url: str) -> datatypes.Points3D:
    """Downloads a PLY point cloud from a URL and loads it as a Points3D object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    point_cloud = io.load_point_cloud(filepath=tmp_path)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded point cloud from {url}")
    return point_cloud


def visualize(point_cloud, projected_point_cloud) -> None:
    """Visualizes the input point cloud, projection plane, and projected point cloud using Rerun."""
    # Initialize Rerun
    rr.init("project_point_cloud_to_plane", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup camera view
    overview_position = np.array([502.75708451, -134.34381185, -519.97747961])
    look_target = np.array([0, 0, 0])
    eye_up = np.array([0.04082638, -0.00847461, -0.99913032])

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
    background = rrb.Background(color=(255, 255, 255))

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Input Point Cloud",
                origin="input_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Projected Point Cloud",
                origin="projected_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))
    # Visualize original point cloud
    rr.log("input_point_cloud/points", rr.Points3D(
        positions=point_cloud.positions,
        colors=point_cloud.colors,
    ))

    # Visualize the projection plane
    plane_coeffs = np.array([0, 0, 1, 0])
    a, b, c, d = plane_coeffs
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize
    point_on_plane = -d / (a**2 + b**2 + c**2) * np.array([a, b, c])

    # Calculate the extent based on the point cloud bounding box
    points_np = np.asarray(point_cloud.positions)
    extent = np.linalg.norm(np.ptp(points_np, axis=0))
    rect_size = 0.3 * extent if extent > 0 else 0.2

    # Create plane rectangle
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    v = np.cross(normal, helper)
    v /= np.linalg.norm(v)

    u = np.cross(normal, v)
    u /= np.linalg.norm(u)

    half = rect_size / 2.0
    rect_corners = np.stack(
        [
            point_on_plane + half * u + half * v,
            point_on_plane - half * u + half * v,
            point_on_plane - half * u - half * v,
            point_on_plane + half * u - half * v,
        ],
        axis=0,
    )

    rect_lines = np.stack([
        np.stack([rect_corners[0], rect_corners[1]]),
        np.stack([rect_corners[1], rect_corners[2]]),
        np.stack([rect_corners[2], rect_corners[3]]),
        np.stack([rect_corners[3], rect_corners[0]]),
    ], axis=0)

    rr.log("input_point_cloud/plane",
           rr.LineStrips3D(rect_lines,
                           radii=rr.Radius.ui_points(3.0),
                           colors=np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (4, 1))))

    rr.log("input_point_cloud/plane_normal",
           rr.Arrows3D(origins=np.array([point_on_plane]),
                       vectors=np.array([normal * extent * 0.2]),
                       colors=np.array([[255, 0, 0]]),
                       radii=2.0))

    # Visualize projected point cloud
    if projected_point_cloud is not None:
        rr.log("projected_point_cloud/points", rr.Points3D(
            positions=projected_point_cloud.positions,
            colors=projected_point_cloud.colors,
        ))


if __name__ == "__main__":
    project_point_cloud_to_plane_example()
