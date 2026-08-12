"""
Demonstrates computing the principal axes of a point cloud using PCA.

This example:
- Downloads an example point cloud.
- Finds the orthogonal axes along which the point cloud has maximum variance.
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


def estimate_principal_axes_example():
    """
    Computes the principal axes of a point cloud using PCA.

    Finds the orthogonal axes along which the point cloud has maximum variance.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_large_pcb_inspection_cropped_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    principal_axes = vitreous.estimate_principal_axes(
        point_cloud=point_cloud,
        method="obb",
    )
    logger.success("Estimated principal axes")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, principal_axes)


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


def visualize(point_cloud, principal_axes) -> None:
    """Visualizes the point cloud with its principal axes using Rerun."""
    # Initialize Rerun
    rr.init("estimate_principal_axes", spawn=False)
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
    overview_position = np.array([250., 375., 250.])
    look_target = np.array([0, 0, 0])
    eye_up = np.array([0., 0., 1.])

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
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Input Point Cloud", origin="input_point_cloud",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
                rrb.Spatial3DView(
                    name="Principal Axes Overlay", origin="principal_axes_overlay",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
            )
        )
    )

    # Log the input point cloud under input_point_cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Calculate the centroid of the point cloud

    # Principal axes is a (3, 3) matrix where each column is an axis (unit vector)
    # Scale the axes to be visible (adjust scale_factor based on point cloud size)
    points = point_cloud.positions
    bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    scale_factor = bbox_size * 0.5  # Scale to 30% of bounding box diagonal

    # Extract the three principal axes (columns) and scale them
    axis1 = principal_axes[:, 0] * scale_factor  # First (largest variance)
    axis2 = principal_axes[:, 1] * scale_factor  # Second
    axis3 = principal_axes[:, 2] * scale_factor  # Third (smallest variance)

    # Log the principal axes as arrows originating from the centroid
    rr.log("principal_axes_overlay/points", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    rr.log("principal_axes_overlay/axes", rr.Arrows3D(
        origins=look_target,
        vectors=axis1,
        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        radii=4))  # RGB for 1st, 2nd, 3rd axes - radii controls arrow thickness


if __name__ == "__main__":
    estimate_principal_axes_example()
