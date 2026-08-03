"""
Demonstrates filtering points within an axis-aligned bounding box.

This example:
- Downloads an example point cloud.
- Keeps only points that fall within the specified 3D box defined by min/max coordinates.
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


def filter_point_cloud_using_bounding_box_example():
    """
    Filters points within an axis-aligned bounding box.

    Keeps only points that fall within the specified 3D box defined by
    min/max coordinates along each axis.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/plastic_2_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    # Create Box
    x_min, y_min, z_min, x_max, y_max, z_max = np.array([-163, -100, 470, 150, 100, 544])
    centers = np.array(
        [[(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]],
        dtype=np.float32,
    )
    half_sizes = np.array(
        [[(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2]],
        dtype=np.float32,
    )
    bbox = datatypes.Boxes3D(half_sizes=half_sizes, centers=centers)

    # Filter point cloud using bounding box
    filtered_point_cloud = vitreous.filter_point_cloud_using_bounding_box(
        point_cloud=point_cloud, bbox=bbox
    )

    logger.success(
        f"Filtered {len(filtered_point_cloud.positions)} points using bounding box"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, filtered_point_cloud, bbox)


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


def visualize(point_cloud, filtered_point_cloud, bbox) -> None:
    """Visualizes the input point cloud, bounding box, and filtered result using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_bounding_box", spawn=False)
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
    look_target = np.array([-16.837628596149877, 12.493554094779665, 516.8440399654662])
    offset = np.array([190.90496669008934, -508.22473543952464, -406.8421301004321])
    camera_eye_position = look_target + offset
    eye_up = np.array([0.17020111661232978, -0.054036235719827574, -0.9839266563789942])

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
    # Log the filtered point cloud with color handling
    rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
           colors=filtered_point_cloud.colors))

    # add the bbox to the point cloud
    rr.log(
        "input_point_cloud/bbox",
        rr.Boxes3D(
            half_sizes=bbox.half_sizes,
            centers=bbox.centers,
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        )
    )


if __name__ == "__main__":
    filter_point_cloud_using_bounding_box_example()
