"""
Demonstrates segmenting the dominant plane from a point cloud using RANSAC.

This example:
- Downloads an example point cloud.
- Finds the largest planar surface in the cloud using random sample consensus and returns inlier points and the plane equation.
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


def segment_point_cloud_using_plane_example():
    """
    Segments the dominant plane from a point cloud using RANSAC.

    Finds the largest planar surface in the cloud using random sample consensus.
    Returns inlier points and plane equation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    segmented_point_cloud, plane_model = vitreous.segment_point_cloud_using_plane(
        distance_threshold=1.0,
        num_initial_points=3,
        max_iterations=1000,
        keep_outliers=False,
        point_cloud=point_cloud,
    )
    logger.success(f"Segmented {len(segmented_point_cloud.positions)} points using plane")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, segmented_point_cloud)


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


def visualize(point_cloud, segmented_point_cloud) -> None:
    """Visualizes the input and segmented point clouds using Rerun."""
    # Initialize Rerun
    rr.init("segment_point_cloud_using_plane", spawn=False)
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
    look_target = np.array([17.017827726026972, -12.119958718903193, 679.1044351589703])
    offset = np.array([482.9731371585466, -460.7716744655184, -394.649557512302])
    position = look_target + offset
    plane_normal = np.array([0.02831009584389102, -0.5750797310951596, -0.817607388271919])
    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position,
        look_target=look_target,
        eye_up=plane_normal,
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
                name="Segmented Point Cloud",
                origin="segmented_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))
    # Visualize input point cloud
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions, colors=point_cloud.colors))
    rr.log("segmented_point_cloud", rr.Points3D(positions=segmented_point_cloud.positions, colors=segmented_point_cloud.colors))


if __name__ == "__main__":
    segment_point_cloud_using_plane_example()
