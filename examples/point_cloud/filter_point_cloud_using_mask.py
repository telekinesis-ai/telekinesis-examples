"""
Demonstrates filtering a structured point cloud using a 2D binary mask.

This example:
- Downloads an example point cloud and a binary mask image.
- Applies the 2D image mask to the organized point cloud, keeping only points where the mask is True.
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


def filter_point_cloud_using_mask_example():
    """
    Filters a structured point cloud using a 2D binary mask.

    Applies a 2D image mask to an organized point cloud, keeping only points
    where the corresponding pixel is True.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_6_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)

    mask_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/can_vertical_6_mask.png"
    mask = fetch_mask(mask_url)
    logger.success("Loaded point cloud and mask")

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.filter_point_cloud_using_mask(
        point_cloud=point_cloud,
        mask=mask,
    )
    logger.success("Filtered points using mask")

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, result_point_cloud, mask)


def fetch_point_cloud(url: str) -> datatypes.Points3D:
    """Downloads a PLY point cloud from a URL and loads it as a Points3D object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    point_cloud = io.load_point_cloud(
        filepath=tmp_path,
        remove_duplicated_points=False,
        remove_infinite_points=False,
        remove_nan_points=False,
    )
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded point cloud from {url}")
    return point_cloud


def fetch_mask(url: str) -> datatypes.Image:
    """Downloads a mask image from a URL and loads it as a grayscale Image object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    mask = io.load_image(filepath=tmp_path, as_gray=True)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded mask from {url}")
    return mask


def visualize(point_cloud, result_point_cloud, mask) -> None:
    """Visualizes the input point cloud, binary mask, and masked result using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_mask", spawn=False)
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
    overview_position = np.array([865, -865, 165])
    look_target = np.array([-9.09364389, -78.71465444, 598.47233982])
    eye_up = np.array([0.02736525, -0.56736208, -0.82301361])

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
                name="Input Point Cloud",
                origin="input_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Binary Mask", origin="binary_mask"),
            ),
            rrb.Spatial3DView(
                name="Masked Point Cloud",
                origin="masked_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Log the input point cloud
    rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Log and mask
    mask_np = mask.to_numpy()
    rr.log("binary_mask", rr.Image(mask_np, color_model="L"))

    # Log the filtered point cloud
    rr.log("masked_point_cloud", rr.ViewCoordinates.RDB, static=True)
    rr.log("masked_point_cloud", rr.Points3D(positions=result_point_cloud.positions,
           colors=result_point_cloud.colors))


if __name__ == "__main__":
    filter_point_cloud_using_mask_example()
