"""
Demonstrates filtering points within axis-aligned min/max ranges.

This example:
- Downloads an example point cloud.
- Keeps only points where each coordinate (x, y, z) falls within specified min/max bounds.
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


def filter_point_cloud_using_pass_through_filter_example():
    """
    Filters points within axis-aligned min/max ranges.

    Keeps only points where each coordinate (x, y, z) falls within specified
    min/max bounds.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/mounts_3_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    x_min, y_min, z_min, x_max, y_max, z_max = np.array([-185.0, -164.0, 450.0, 230.0, 164.0, 548.0])

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_pass_through_filter(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        point_cloud=point_cloud,
    )
    logger.success("Filtered points using axis-aligned range")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, filtered_point_cloud, x_min, y_min, z_min, x_max, y_max, z_max)


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


def visualize(point_cloud, filtered_point_cloud, x_min, y_min, z_min, x_max, y_max, z_max) -> None:
    """Visualizes the input point cloud, filter box, and filtered result using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_pass_through_filter", spawn=False)
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
    look_target = np.array([17.246607005843458, -10.312582127251696, 495.8964079473356])
    offset = np.array([640.2690590947808, -332.28717547581937, -727.8142110040502])
    camera_eye_position = look_target + offset
    eye_up = np.array([0.040600170502887244, 0.009404387964181355, -0.9991312144268918])

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

    # Log the input point cloud
    rr.log("input_point_cloud/points", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Log the passthrough filter box on the same view
    box_corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])
    box_lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    rr.log("input_point_cloud/filter_box", rr.LineStrips3D([box_corners[line]
           for line in box_lines], colors=np.array([[255, 0, 0]])))

    # Log the filtered point cloud with color handling
    rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
           colors=filtered_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_pass_through_filter_example()
