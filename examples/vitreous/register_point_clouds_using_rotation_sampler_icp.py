"""
Demonstrates finding the best alignment by trying multiple rotations with ICP refinement.

This example:
- Downloads two example point clouds (source and target).
- Samples rotations in Euler angle space, runs ICP for each, and keeps the best.
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


def register_point_clouds_using_rotation_sampler_icp_example():
    """
    Finds best alignment by trying multiple rotations with ICP refinement.

    Samples rotations in Euler angle space, runs ICP for each, and keeps the best.
    """
    # ===================== Load Data ==========================================
    source_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_bottle_cylinder_centered.ply"
    target_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_bottle_segmented.ply"
    source_point_cloud = fetch_point_cloud(source_url)
    target_point_cloud = fetch_point_cloud(target_url)
    logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_rotation_sampler_icp(
        x_step_size_deg=30,
        y_step_size_deg=10,
        z_step_size_deg=30,
        x_min_deg=0,
        x_max_deg=180,
        y_min_deg=0,
        y_max_deg=180,
        z_min_deg=0,
        z_max_deg=180,
        early_stop_fitness_score=0.9,
        min_fitness_score=0.2,
        max_iterations=100,
        max_correspondence_distance=2,
        estimate_scaling=False,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )
    logger.success("Registered point clouds using rotation sampler ICP")

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
    rr.init("register_point_clouds_using_rotation_sampler_icp", spawn=False)
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
    overview_position = np.array([356.32651595, 468.61847702, 563.96842436])
    look_target = np.array([-169.68149308, -57.38953201, 37.96041533])
    eye_up = np.array([0.0, 0.0, 1.0])

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

    # After: Show aligned result (red) and target (green) overlapping
    rr.log("after_registration/source_aligned", rr.Points3D(
        positions=aligned_source.positions,
        colors=[[255, 0, 0]] * len(aligned_source.positions),
    ))
    rr.log("after_registration/target", rr.Points3D(
        positions=target_point_cloud.positions,
        colors=[[0, 255, 0]] * len(target_point_cloud.positions),
    ))


if __name__ == "__main__":
    register_point_clouds_using_rotation_sampler_icp_example()
