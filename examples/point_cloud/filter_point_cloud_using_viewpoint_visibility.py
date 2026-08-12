"""
Demonstrates filtering points based on visibility from a camera viewpoint.

This example:
- Downloads an example point cloud.
- Removes points that are occluded or outside the visibility range from a specified camera position.
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


def filter_point_cloud_using_viewpoint_visibility_example():
    """
    Filters points based on visibility from a camera viewpoint.

    Removes points that are occluded or outside the visibility range from
    a specified camera position.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_parcels_04_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_viewpoint_visibility(
        viewpoint=[100, -500, 250.0],
        visibility_radius=100000.0,
        point_cloud=point_cloud,
    )
    logger.success("Filtered points using viewpoint visibility")

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
    """Visualizes the input and viewpoint-filtered point clouds using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_viewpoint_visibility", spawn=False)
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
    overview_position = np.array([-180., -600.,  660.])
    look_target = np.array([0., 0., 0.])
    eye_up = np.array([0., 0., 1.])
    camera_eye_position = np.array([ 100., -500.,  250.])


    # EyeControls:
    # - overview_eye_controls for View 1 & 3 (zoomed out)
    # - camera_eye_controls for View 2 (exact viewpoint)
    overview_eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=overview_position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.5,
        speed=0.0,
    )

    camera_eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=camera_eye_position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.0,
        speed=0.0,
    )

    # Send blueprint
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Input (Centered) – Overview (zoomed out)",
                    origin="input_point_cloud",
                    background=background,
                    eye_controls=overview_eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information,
                ),
                rrb.Spatial3DView(
                    name="Camera View – What the camera sees",
                    origin="input_point_cloud",
                    background=background,
                    eye_controls=camera_eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information,
                ),
                rrb.Spatial3DView(
                    name="Filtered (Centered) – Overview (zoomed out)",
                    origin="output_point_cloud",
                    background=background,
                    eye_controls=overview_eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information,
                ),
            )
        )
    )

    # Center the point clouds for better visualization
    rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
    rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

    # Log centered input
    rr.log(
            "input_point_cloud",
            rr.Points3D(
                positions=point_cloud.positions,
                colors=point_cloud.colors,
            ),
        )

    # Log centered filtered cloud
    rr.log(
            "output_point_cloud",
            rr.Points3D(
                positions=filtered_point_cloud.positions,
                colors=filtered_point_cloud.colors,
            ),
        )

    # Show the camera location as a red dot
    rr.log(
        "camera_viewpoint",
        rr.Points3D(
            positions=overview_position,
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
        ),
    )


if __name__ == "__main__":
    filter_point_cloud_using_viewpoint_visibility_example()
