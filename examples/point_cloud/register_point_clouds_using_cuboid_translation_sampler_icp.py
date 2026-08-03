"""
Demonstrates finding the best alignment by sampling translations in a 3D grid (cuboid) with ICP.

This example:
- Downloads two example point clouds (source and target).
- Tries translations on a regular 3D grid, runs ICP for each, and keeps the best result.
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


def register_point_clouds_using_cuboid_translation_sampler_icp_example():
    """
    Finds best alignment by sampling translations in a 3D grid (cuboid) with ICP.

    Tries translations on a regular 3D grid within specified x/y/z ranges, runs ICP
    for each, and keeps best result.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/weld_clamp_model_shifted.ply"
    target_point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/weld_clamp_cluster_0_centroid_registered.ply"
    source_point_cloud = fetch_point_cloud(source_point_cloud_url)
    target_point_cloud = fetch_point_cloud(target_point_cloud_url)
    logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    transformation_matrix = (
        vitreous.register_point_clouds_using_cuboid_translation_sampler_icp(
            step_size=2,
            x_min=-20,
            x_max=20,
            y_min=-20,
            y_max=20,
            z_min=-20,
            z_max=20,
            early_stop_fitness_score=0.3,
            min_fitness_score=0.48,
            max_iterations=50,
            max_correspondence_distance=2,
            estimate_scaling=False,
            source_point_cloud=source_point_cloud,
            target_point_cloud=target_point_cloud,
            initial_transformation_matrix=np.eye(4),
        )
    )
    logger.success(f"Registered point clouds using cuboid translation sampler ICP: transformation_matrix: {transformation_matrix.matrix}")

    # ===================== Visualization  (Optional) ======================
    visualize(source_point_cloud, target_point_cloud, transformation_matrix)


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


def visualize(source_point_cloud, target_point_cloud, transformation_matrix) -> None:
    """Visualizes the point clouds before and after registration using Rerun."""
    # Initialize Rerun
    rr.init("register_point_clouds_using_cuboid_translation_sampler_icp", spawn=False)
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
    overview_position = np.array([182.91228535, 149.88950409, 796.55904478])
    look_target = np.array([-60.82915742, -93.85193868, 552.81760201])
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
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Before Registration",
                origin="before_registration",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="After Registration",
                origin="after_registration",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Create aligned source point cloud
    aligned_source = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix.matrix,
        modify_inplace=False,
    )

    # Before: Show source (red) and target (green) misaligned
    rr.log("before_registration/source", rr.Points3D(
        positions=source_point_cloud.positions,
        colors=[[255, 0, 0]] * len(source_point_cloud.positions),
    ))
    rr.log("before_registration/target", rr.Points3D(
        positions=target_point_cloud.positions,
        colors=[[0, 255, 0]] * len(target_point_cloud.positions),
    ))

    # After: Show aligned source (red) and target (green) overlapping
    rr.log("after_registration/source_aligned", rr.Points3D(
        positions=aligned_source.positions,
        colors=[[255, 0, 0]] * len(aligned_source.positions),
    ))
    rr.log("after_registration/target", rr.Points3D(
        positions=target_point_cloud.positions,
        colors=[[0, 255, 0]] * len(target_point_cloud.positions),
    ))


if __name__ == "__main__":
    register_point_clouds_using_cuboid_translation_sampler_icp_example()
