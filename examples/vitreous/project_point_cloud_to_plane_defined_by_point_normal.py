"""
Demonstrates projecting a point cloud onto a plane defined by a point and normal.

This example:
- Downloads an example point cloud.
- Projects points onto a plane defined by a point and normal (alternative parameterization).
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


def project_point_cloud_to_plane_defined_by_point_normal_example():
    """
    Projects points onto a plane defined by a point and normal (alternative parameterization).

    Same as plane projection but using point+normal instead of coefficients.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])

    projected_point_cloud = vitreous.project_point_cloud_to_plane_defined_by_point_normal(
        add_white_noise=False,
        white_noise_standard_deviation=1e-6,
        point_cloud=point_cloud,
        point=plane_point,
        plane_normal=plane_normal,
    )
    logger.success(
        f"Projected {len(projected_point_cloud.positions)} points to plane (point+normal)"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, projected_point_cloud, plane_point, plane_normal)


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


def visualize(point_cloud, projected_point_cloud, plane_point, plane_normal) -> None:
    """Visualizes the input point cloud, projection plane, and projected point cloud using Rerun."""
    # Initialize Rerun
    rr.init("project_point_cloud_to_plane_defined_by_point_normal", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup additional rerun settings
    line_grid = rrb.LineGrid3D(visible=False)
    spatial_information = rrb.SpatialInformation(
        target_frame="tf#/",
        show_axes=False,
        show_bounding_box=False,
    )
    background = rrb.Background(color=(255, 255, 255))

    # Setup camera view
    overview_position = np.array([502.75708451, -134.34381185, -519.97747961])
    look_target = np.array([0, 0, 0])
    eye_up = np.array([0.04082638, -0.00847461, -0.99913032])

    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=overview_position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.5,
        speed=0.0,
        tracking_entity=None,
    )

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Input Point Cloud",
                origin="input_point_cloud_and_plane",
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
    # Log input point cloud
    rr.log("input_point_cloud_and_plane/points", rr.Points3D(
        positions=point_cloud.positions,
        colors=point_cloud.colors,
    ))

    # Log plane
    points_np = np.asarray(point_cloud.positions)
    extent = np.linalg.norm(np.ptp(points_np, axis=0))
    rect_size = 0.3 * extent if extent > 0 else 0.2

    # Create plane rectangle
    helper = np.array([0.0, 0.0, 1.0]) if abs(plane_normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    v = np.cross(plane_normal, helper)
    v /= np.linalg.norm(v)

    u = np.cross(plane_normal, v)
    u /= np.linalg.norm(u)

    half = rect_size / 2.0
    rect_corners = np.stack(
        [
            plane_point + half * u + half * v,
            plane_point - half * u + half * v,
            plane_point - half * u - half * v,
            plane_point + half * u - half * v,
        ],
        axis=0,
    )
    rect_lines = np.stack(
        [
            np.stack([rect_corners[0], rect_corners[1]]),
            np.stack([rect_corners[1], rect_corners[2]]),
            np.stack([rect_corners[2], rect_corners[3]]),
            np.stack([rect_corners[3], rect_corners[0]]),
        ],
        axis=0,
    )

    rr.log(
        "input_point_cloud_and_plane/plane",
        rr.LineStrips3D(
            rect_lines,
            radii=rr.Radius.ui_points(3.0),  # constant screen-space thickness
            colors=np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (4, 1)),
        ),
    )

    # Log plane normal vector
    normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    rr.log("input_point_cloud_and_plane/plane_normal",
           rr.Arrows3D(origins=np.array([plane_point]),
                       vectors=np.array([normal_normalized * extent * 0.2]),
                       colors=np.array([[255, 0, 0]]),
                       radii=2.0))

    # Log projected point cloud
    rr.log("projected_point_cloud/points", rr.Points3D(
        positions=projected_point_cloud.positions,
        colors=projected_point_cloud.colors,
    ))


if __name__ == "__main__":
    project_point_cloud_to_plane_defined_by_point_normal_example()
