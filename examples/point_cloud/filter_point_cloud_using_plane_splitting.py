"""
Demonstrates splitting a point cloud by a plane, keeping one side.

This example:
- Downloads an example point cloud.
- Divides a point cloud using a plane and keeps points on either the positive or negative side.
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


def filter_point_cloud_using_plane_splitting_example():
    """
    Splits a point cloud by a plane, keeping one side.

    Divides a point cloud using a plane and keeps points on either the positive
    or negative side.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/mounts_3_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_plane_splitting(
        keep_positive_side=False,
        point_cloud=point_cloud,
        plane_coefficients=[0, 0, 1, -547],
    )
    logger.success("Filtered points using plane splitting")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, filtered_point_cloud)


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


def visualize(point_cloud, filtered_point_cloud) -> None:
    """Visualizes the input and output point clouds using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_plane_splitting", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup camera view
    overview_position = np.array([ 433.57570131, -301.27080108, -26.4841608 ])
    look_target = np.array([ 10.9282587, -9.28041238, 493.51069811])
    eye_up = np.array([ 0.04087094, 0.0086678, -0.99912684])


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

    # Set the blueprint panel for 3D point cloud visualization (input left, output right, vertical)
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
                name="Output Point Cloud",
                origin="output_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Log the input point cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
               colors=point_cloud.colors))

    # Log the output point cloud
    rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
               colors=filtered_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_plane_splitting_example()
