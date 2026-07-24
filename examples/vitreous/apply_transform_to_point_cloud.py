"""
Demonstrates applying a 6-DOF rigid transformation (rotation + translation) to a point cloud.

This example:
- Downloads an example point cloud.
- Transforms points using a 4x4 homogeneous transformation matrix.
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


def apply_transform_to_point_cloud_example():
    """
    Applies a 6-DOF rigid transformation (rotation + translation) to a point cloud.

    Transforms points using a 4x4 homogeneous transformation matrix.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/plastic_centered.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    transformed_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=point_cloud,
        transformation_matrix= [[1, 0, 0, 15], [0, 1, 0, 15], [0, 0, 1, 5], [0, 0, 0, 1]],
        modify_inplace=False
    )
    logger.success(f"Applied transform to {len(transformed_point_cloud.positions)} points")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, transformed_point_cloud)


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


def visualize(point_cloud, transformed_point_cloud) -> None:
    """Visualizes the source and transformed point clouds using Rerun."""
    # Initialize Rerun
    rr.init("apply_transform_to_point_cloud", spawn=False)
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
    overview_position = np.array([ 51.87143098,   2.90578544, -47.97485367])
    look_target = np.array([0, 0, 0])
    eye_up = np.array([ 0.03973926,  0.00298701, -0.99920562])

    eye_controls_original = rrb.EyeControls3D(
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
                name="Source Point Cloud",
                origin="source_point_cloud",
                background=background,
                eye_controls=eye_controls_original,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Transformed Point Cloud",
                origin="transformed_point_cloud",
                background=background,
                eye_controls=eye_controls_original,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Visualize original point cloud
    rr.log("source_point_cloud", rr.Points3D(
        positions=point_cloud.positions,
        colors=point_cloud.colors)
       )

    # Draw origin frame for source point cloud
    axis_length = 10  # Adjust based on point cloud scale
    rr.log("source_point_cloud/origin_frame", rr.Arrows3D(
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for X, Y, Z
    ))

    # Visualize transformed point cloud
    if transformed_point_cloud is not None:
        rr.log("transformed_point_cloud", rr.Points3D(
            positions=transformed_point_cloud.positions,
            colors=transformed_point_cloud.colors)
           )

        # Draw origin frame for transformed point cloud
        rr.log("transformed_point_cloud/origin_frame", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for X, Y, Z
        ))
    else:
        logger.error("Transformation failed: No transformed point cloud to log.")


if __name__ == "__main__":
    apply_transform_to_point_cloud_example()
