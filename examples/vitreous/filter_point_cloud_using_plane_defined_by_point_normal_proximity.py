"""
Demonstrates filtering points near a plane defined by a point and normal vector.

This example:
- Downloads an example point cloud.
- Keeps points within a distance threshold of a plane specified by a point on the plane and its normal vector.
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


def filter_point_cloud_using_plane_defined_by_point_normal_proximity_example():
    """
    Filters points near a plane defined by a point and normal vector.

    Keeps points within a distance threshold of a plane specified by a point
    on the plane and its normal vector.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = (
        vitreous.filter_point_cloud_using_plane_defined_by_point_normal_proximity(
            distance_threshold=4.0,
            point_cloud=point_cloud,
            plane_point=[-15.74520074, 319.25105712, 454.3114797],
            plane_normal=[0.028344755192329624, -0.5747207168510667, -0.8178585895344518],
        )
    )
    logger.success("Filtered points using plane defined by point and normal")

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
    """Visualizes the input and filtered point clouds using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_plane_defined_by_point_normal_proximity", spawn=False)
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
    overview_position = np.array([ 508.26353625, -457.76445726,  289.53896696])
    look_target = np.array([ 21.82833776,  -6.47603561, 684.12881138])
    eye_up = np.array([ 0.02846775, -0.57502007, -0.81764387])


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
                origin="filtered_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))


    # Log the input point cloud under input_point_cloud + plane
    rr.log("input_point_cloud", rr.Points3D(point_cloud.positions,
               colors=point_cloud.colors))

    # Log the output point cloud
    rr.log("filtered_point_cloud", rr.Points3D(np.asarray(filtered_point_cloud.positions),
               colors=filtered_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_plane_defined_by_point_normal_proximity_example()
