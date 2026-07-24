"""
Demonstrates scaling a point cloud uniformly about a center point.

This example:
- Downloads an example point cloud.
- Multiplies all point coordinates by a scale factor relative to a center.
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


def scale_point_cloud_example():
    """
    Scales a point cloud uniformly about a center point.

    Multiplies all point coordinates by a scale factor relative to a center.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/relay_2_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    scaled_point_cloud = vitreous.scale_point_cloud(
        point_cloud=point_cloud,
        center_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale_factor=0.3,
        modify_inplace=False)
    logger.success(f"Scaled point cloud to {len(scaled_point_cloud.positions)} points")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, scaled_point_cloud)


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


def visualize(point_cloud, scaled_point_cloud) -> None:
    """Visualizes the input and scaled point clouds using Rerun."""
    # Initialize Rerun
    rr.init("scale_point_cloud", spawn=False)
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
    overview_position = np.array([117.44420607, -90.56381865, 110.3344537])
    look_target = np.array([18.521805, 4.61124328, 282.22516171])
    eye_up = np.array([-0.07754533, 0.34059355, -0.93700734])

    eye_controls_original = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=overview_position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.5,
        speed=0.0,
        tracking_entity=None,
    )

    look_target = np.array([5.556541501027933, 1.383372985065839, 84.66754851275184])
    offset = np.array([80.57584366, -106.54735255, -171.92510329])
    camera_eye_position = look_target + offset
    eye_up = np.array([-0.0774216, 0.34095774, -0.93688511])

    eye_controls_scaled= rrb.EyeControls3D(
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
                origin="original_point_cloud",
                background=background,
                eye_controls=eye_controls_original,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Scaled Point Cloud",
                origin="scaled_point_cloud",
                background=background,
                eye_controls=eye_controls_scaled,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Visualize original point cloud
    rr.log("original_point_cloud", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Visualize scaled point cloud
    if scaled_point_cloud is not None:
        rr.log("scaled_point_cloud", rr.Points3D(positions=scaled_point_cloud.positions,
               colors=scaled_point_cloud.colors))
    else:
        logger.error("Scaling failed: No scaled point cloud to log.")


if __name__ == "__main__":
    scale_point_cloud_example()
