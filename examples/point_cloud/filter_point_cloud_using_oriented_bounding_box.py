"""
Demonstrates filtering points within an oriented (rotated) bounding box.

This example:
- Downloads an example point cloud.
- Keeps only points within a 3D box that can be rotated to any orientation.
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


def filter_point_cloud_using_oriented_bounding_box_example():
    """
    Filters points within an oriented (rotated) bounding box.

    Keeps only points within a 3D box that can be rotated to any orientation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    x_min = -205.65248652
    y_min = -112.59310319
    z_min = 554.42936219
    x_max = 121.88022318
    y_max = -17.60647882
    z_max = 698.54912862
    rot_x = -38.1245801
    rot_y = -7.89877607
    rot_z = -7.74440359

    half_sizes = np.array(
        [[(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2]],
        dtype=np.float32,
    )
    centers = np.array(
        [[(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]],
        dtype=np.float32,
    )
    rotations_in_euler_angle = np.array([[rot_x, rot_y, rot_z]], dtype=np.float32)
    oriented_bbox = datatypes.Boxes3D(
        half_sizes=half_sizes,
        centers=centers,
        rotations_in_euler_angle=rotations_in_euler_angle,
    )

    # Filter point cloud using oriented bounding box
    filtered_point_cloud = vitreous.filter_point_cloud_using_oriented_bounding_box(
        point_cloud=point_cloud, oriented_bbox=oriented_bbox
    )
    logger.success("Filtered points using oriented bounding box")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, filtered_point_cloud, oriented_bbox)


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


def visualize(point_cloud, filtered_point_cloud, oriented_bbox) -> None:
    """Visualizes the input and filtered point clouds with the oriented bounding box using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_oriented_bounding_box", spawn=False)
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
    look_target = np.array([-39.32096224866192, -77.51841497655289, 602.2689849331848])
    offset = np.array([587.536048681736, -530.2253280469823, -473.4666019099898])
    camera_eye_position = look_target + offset
    eye_up = np.array([0.02844979765608562, -0.5751413943408177, -0.8175591633203237])

    logger.success(f"Camera eye position: {camera_eye_position}, look target: {look_target}, eye up: {eye_up}")

    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=camera_eye_position,
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

    # Log the input point cloud under input_point_cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
                                            colors=point_cloud.colors))

    # Log the output point cloud under output_point_cloud
    rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
                                               colors=filtered_point_cloud.colors))

    # Log the oriented bounding box on the input view
    quaternions = R.from_euler('xyz', oriented_bbox.rotations_in_euler_angle, degrees=True).as_quat()
    rr.log("input_point_cloud/oriented_bbox", rr.Boxes3D(
        half_sizes=oriented_bbox.half_sizes,
        centers=oriented_bbox.centers,
        quaternions=quaternions,
        colors=[(255, 0, 0)],
    ))


if __name__ == "__main__":
    filter_point_cloud_using_oriented_bounding_box_example()
