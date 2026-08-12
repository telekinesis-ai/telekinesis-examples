"""
Demonstrates removing points from one cloud that are near points in another cloud.

This example:
- Downloads two example point clouds.
- Subtracts point_cloud2 from point_cloud1 by removing any point in cloud1 within a distance threshold of any point in cloud2.
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


def subtract_point_clouds_example():
    """
    Removes points from one cloud that are near points in another cloud.

    Subtracts point_cloud2 from point_cloud1 by removing any point in cloud1
    that is within distance_threshold of any point in cloud2.
    """
    # ===================== Load Data ==========================================
    point_cloud_url_1 = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_mixed_grocery_pallet_centered.ply"
    point_cloud_url_2 = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_mixed_grocery_pallet_box_filtered.ply"
    point_cloud1 = fetch_point_cloud(point_cloud_url_1)
    point_cloud2 = fetch_point_cloud(point_cloud_url_2)
    logger.success(f"Loaded point cloud 1 with {len(point_cloud1.positions)} points")
    logger.success(f"Loaded point cloud 2 with {len(point_cloud2.positions)} points")

    # ===================== Run Skill ==========================================
    subtracted_point_cloud = vitreous.subtract_point_clouds(
        distance_threshold=0.1,
        point_cloud1=point_cloud1,
        point_cloud2=point_cloud2,
    )
    logger.success("Subtracted point clouds")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud1, point_cloud2, subtracted_point_cloud)


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


def visualize(point_cloud1, point_cloud2, subtracted_point_cloud) -> None:
    """Visualizes the two input point clouds and the subtracted result using Rerun."""
    # Initialize Rerun
    rr.init("subtract_point_clouds", spawn=False)
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
    overview_position = np.array([ 100, 1500, 2000])
    look_target = np.array([100, 0, 0])
    eye_up = np.array([0, 0, 1])

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
            rrb.Vertical(
                rrb.Spatial3DView(
                    name="Point Cloud 1",
                    origin="point_cloud_1",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
                rrb.Spatial3DView(
                    name="Point Cloud 2",
                    origin="point_cloud_2",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
            ),
            rrb.Spatial3DView(
                name="Subtracted Point Cloud",
                origin="subtracted_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information
            ),)
    ))

    # Log the first point cloud under point_cloud_1
    rr.log("point_cloud_1", rr.Points3D(positions=point_cloud1.positions,
               colors=point_cloud1.colors))

    # Log the second point cloud under point_cloud_2
    rr.log("point_cloud_2", rr.Points3D(positions=point_cloud2.positions,
           colors=point_cloud2.colors))

    # Log the subtracted point cloud under subtracted_point_cloud
    rr.log("subtracted_point_cloud", rr.Points3D(positions=subtracted_point_cloud.positions,
           colors=subtracted_point_cloud.colors))


if __name__ == "__main__":
    subtract_point_clouds_example()
