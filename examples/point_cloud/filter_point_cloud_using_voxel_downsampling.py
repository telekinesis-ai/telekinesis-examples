"""
Demonstrates downsampling a point cloud using voxel grid averaging.

This example:
- Downloads an example point cloud.
- Divides 3D space into voxels and replaces all points within each voxel with their centroid.
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


def filter_point_cloud_using_voxel_downsampling_example():
    """
    Downsamples a point cloud using voxel grid averaging.

    Divides 3D space into voxels and replaces all points within each voxel
    with their centroid.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_1_subtracted.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_voxel_downsampling(
        voxel_size=0.005, point_cloud=point_cloud
    )
    logger.success("Filtered points using voxel downsampling")

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
    """Visualizes the input and voxel-downsampled point clouds using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_voxel_downsampling", spawn=False)
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
    overview_position = np.array([-0.17948, -0.00342, 0.57515])
    eye_up = np.array([-0.0003322,-0.0688132,-0.1021320])
    look_target = np.array([ 0.01602358, -0.131963, 0.66461711])

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
                origin="input_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Filtered Point Cloud",
                origin="output_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
    rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

    # Log the input point cloud under input_point_cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions, colors=point_cloud.colors))

    # Log the output point cloud under output_point_cloud
    rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions, colors=filtered_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_voxel_downsampling_example()
