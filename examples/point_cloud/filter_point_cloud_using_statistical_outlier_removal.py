"""
Demonstrates removing statistical outliers based on distance distribution.

This example:
- Downloads an example point cloud.
- Removes points that are farther than a threshold from their neighbors, where the threshold is computed from mean distance and standard deviation.
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


def filter_point_cloud_using_statistical_outlier_removal_example():
    """
    Removes statistical outliers based on distance distribution.

    Removes points that are farther than a threshold from their neighbors,
    where the threshold is computed from mean distance and standard deviation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_6_masked.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_statistical_outlier_removal(
        num_neighbors=90,
        standard_deviation_ratio=0.1,
        point_cloud=point_cloud,
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud.positions)} points using statistical outlier removal")

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
    rr.init("filter_point_cloud_using_statistical_outlier_removal", spawn=False)
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
    overview_position = np.array([ 191.10105334, -405.66455294,  458.89275463])
    look_target = np.array([ -9.34432069, -78.6523904,  597.00921687])
    eye_up = np.array([ 0.02866881, -0.56233476, -0.82641256])

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
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
               colors=point_cloud.colors))

    # Log the output point cloud under output_point_cloud
    rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
           colors=filtered_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_statistical_outlier_removal_example()
