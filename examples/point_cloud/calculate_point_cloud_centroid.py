"""
Demonstrates computing the geometric center (centroid) of a point cloud.

This example:
- Downloads an example point cloud.
- Calculates the mean position of all points in the cloud.
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


def calculate_point_cloud_centroid_example():
    """
    Computes the geometric center (centroid) of a point cloud.

    Calculates the mean position of all points in the cloud.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_large_pcb_inspection_cropped_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    centroid = vitreous.calculate_point_cloud_centroid(point_cloud=point_cloud)
    logger.success(
        f"Calculated centroid {centroid} for {len(point_cloud.positions)} points"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, centroid)


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


def visualize(point_cloud, centroid) -> None:
    """Visualizes the point cloud with its centroid using Rerun."""
    # Initialize Rerun
    rr.init("calculate_point_cloud_centroid", spawn=False)
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
    overview_position = np.array([250., 375., 250.])
    look_target = np.array([0, 0, 0])
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
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Input Point Cloud (Box-Filtered)",
                    origin="input",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
                rrb.Spatial3DView(
                    name="Base Points + Frames Overlay",
                    origin="output",
                    background=background,
                    eye_controls=eye_controls,
                    line_grid=line_grid,
                    spatial_information=spatial_information
                ),
            )
        )
    )

    # Log input point cloud
    rr.log("input", rr.ViewCoordinates.RDB, static=True)
    rr.log("input", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Log output point cloud and centroid
    rr.log("output", rr.ViewCoordinates.RDB, static=True)
    # Object cloud
    rr.log("output/object", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))
    # Centroid point
    rr.log("output/centroid", rr.Points3D(positions=centroid, colors=(255, 0, 0)))

    # Log world-aligned frame axes (identity orientation)
    frame_scale = 100
    x_axis = np.array([frame_scale, 0, 0])
    y_axis = np.array([0, frame_scale, 0])
    z_axis = np.array([0, 0, frame_scale])

    axes_single = np.stack([x_axis, y_axis, z_axis], 0)
    axis_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    origins = np.repeat(centroid.reshape(1, 3), 3, axis=0)
    vectors = np.tile(axes_single, (1, 1))
    colors = np.tile(axis_colors, (1, 1))

    rr.log("output/base_frames", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


if __name__ == "__main__":
    calculate_point_cloud_centroid_example()
