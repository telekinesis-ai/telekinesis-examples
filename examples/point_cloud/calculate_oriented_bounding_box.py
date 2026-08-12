"""
Demonstrates computing the oriented bounding box (OBB) of a point cloud.

This example:
- Downloads an example point cloud.
- Finds the smallest box (in any orientation) that contains all points.
- Visualizes the result using Rerun.
"""

import pathlib
import tempfile

import numpy as np
import requests
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb
from scipy.spatial.transform import Rotation as R

from datatypes import datatypes, io
from telekinesis import vitreous


def calculate_oriented_bounding_box_example():
    """
    Computes the oriented bounding box (OBB) of a point cloud.

    Finds the smallest box (in any orientation) that contains all points.
    Returns box parameters including center, extents, and rotation angles.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_1_raw_obb_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    result_bbox = vitreous.calculate_oriented_bounding_box(point_cloud=point_cloud)
    logger.success(
        f"Calculated oriented bounding box for {len(point_cloud.positions)} points with half-size: {result_bbox.half_sizes}, centers: {result_bbox.centers}, rotations_in_euler_angle: {result_bbox.rotations_in_euler_angle}"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, result_bbox)


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


def visualize(point_cloud, result_bbox) -> None:
    """Visualizes the point cloud with its oriented bounding box using Rerun."""
    # Initialize Rerun
    rr.init("calculate_oriented_bounding_box", spawn=False)
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
    overview_position = np.array([ 1800.43642, 0.23659945, -500.517691])
    look_target = np.array([ 0.43641739, 0.23659945, -0.51769066])
    eye_up = np.array([ 0., 0., -1.])

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
                    name="Input Point Cloud",
                    origin="input_point_cloud",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
                rrb.Spatial3DView(
                    name="Oriented Bounding Box Overlay",
                    origin="obb_overlay",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
            )
        )
    )

    # Log the oriented bounding box as a box and overlay it on the point cloud
    # Log the input point cloud under input_point_cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))
    rr.log("obb_overlay", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Log the oriented bounding box
    quaternions = R.from_euler('xyz', result_bbox.rotations_in_euler_angle, degrees=True).as_quat()
    rr.log("obb_overlay", rr.Boxes3D(
        half_sizes=result_bbox.half_sizes,
        centers=result_bbox.centers,
        colors=np.array([[0, 255, 0]]),  # Green color for bounding box
        quaternions=quaternions
    ))


if __name__ == "__main__":
    calculate_oriented_bounding_box_example()
