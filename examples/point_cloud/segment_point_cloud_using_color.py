"""
Demonstrates segmenting points by color similarity to a target color.

This example:
- Downloads an example point cloud.
- Keeps points whose RGB color is within a distance threshold (Euclidean in RGB space) of a target color.
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


def segment_point_cloud_using_color_example():
    """
    Segments points by color similarity to a target color.

    Keeps points whose RGB color is within a distance threshold (Euclidean in
    RGB space) of a target color.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    segmented_point_cloud = vitreous.segment_point_cloud_using_color(
        target_color=[50, 75, 200],
        color_distance_threshold=60.0,
        point_cloud=point_cloud,
    )
    logger.success(f"Segmented {len(segmented_point_cloud.positions)} points using color")

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
    rr.init("segment_point_cloud_using_color", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup camera view
    look_target = np.array([-27.917760217865144, 8.154586928673055, 529.1368178181901])
    offset = np.array([490.3580603260833, -175.19232461052903, -520.3408869973264])
    position = look_target + offset
    eye_up = np.array([0.04159355788852604, -0.009328899120838747, -0.9990910607063638])


    # Add EyeControls3D with all parameters for camera movement tuning
    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,  # Camera control type: Orbital or FirstPerson
        position=position,  # Initial camera position (None = auto)
        look_target=look_target,  # Point the camera looks at (None = auto)
        eye_up=eye_up,  # Up direction vector (None = auto)
        spin_speed=0.5,  # Speed of camera rotation/spin
        speed=0.0,  # Translation speed of camera movement
        tracking_entity=None,  # Entity to track (None = no tracking)
    )

    line_grid = rrb.LineGrid3D(
        visible=False,  # The grid is enabled by default, but you can hide it with this property.
    )

    spatial_information = rrb.SpatialInformation(
        target_frame="tf#/",
        show_axes=False,
        show_bounding_box=False,
    )
    background = rrb.Background(color=(255, 255, 255))

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
    rr.log("input_point_cloud", rr.Points3D(
        positions=point_cloud.positions,
        colors=point_cloud.colors
    ))

    # Visualize segmented point cloud
    rr.log("segmented_point_cloud", rr.Points3D(
        positions=segmented_point_cloud.positions,
        colors=segmented_point_cloud.colors
    ))


if __name__ == "__main__":
    segment_point_cloud_using_color_example()
