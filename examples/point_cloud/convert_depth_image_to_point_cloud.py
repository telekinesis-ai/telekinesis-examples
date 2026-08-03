"""
Demonstrates converting a depth image to a point cloud using camera intrinsics.

This example:
- Downloads an example depth image.
- Converts the depth values to metres.
- Projects the image into a 3D point cloud using camera intrinsics.
- Visualizes the depth image and point cloud using Rerun.
"""

import pathlib
import sys
import tempfile
import types

import numpy as np
import requests
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from datatypes import datatypes, io
from telekinesis import vitreous


def convert_depth_image_to_point_cloud_example():
    """Converts a metric depth image to a point cloud using camera intrinsics."""
    # ===================== Load Data ==========================================
    depth_image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/depth_images/bin_picking.png"
    depth_image = fetch_depth_image(depth_image_url)
    logger.success(f"Loaded image from {depth_image_url}")

    # ===================== Run Skill ==========================================
    # Define the camera intrinsic matrix
    intrinsic_matrix = np.array(
        [
            [643.9369506835938, 0.0, 644.6060791015625],
            [0.0, 643.9369506835938, 354.50860595703125],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    point_cloud = vitreous.convert_depth_image_to_point_cloud(
        depth_image=depth_image,
        intrinsic_matrix=intrinsic_matrix,
    )
    logger.success(
        f"Converted depth image to "
        f"{len(point_cloud.positions)} points"
    )

    # ===================== Visualization (Optional) ===========================
    visualize(depth_image, point_cloud)


def fetch_depth_image(url: str) -> datatypes.Image:
    """Downloads a depth image from a URL and loads it as an Image object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    depth_image = io.load_image(filepath=tmp_path)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded depth image from {url}")
    return depth_image


def visualize(depth_image: datatypes.Image, point_cloud: datatypes.Points3D) -> None:
    """Visualizes the input depth image and output point cloud using Rerun."""
    depth_array = depth_image.to_numpy()

    # Scale the point cloud to fit within the Rerun visualization space
    scaled_point_cloud = vitreous.scale_point_cloud(
        point_cloud=point_cloud,
        scale_factor=0.001,
        center_point=np.array([0.0, 0.0, 0.0]),
    )

    # Apply a transformation to the point cloud to adjust its orientation and position
    transformation_matrix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.8660254, -0.5, 0.2],
            [0.0, -0.5, -0.8660254, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=scaled_point_cloud,
        transformation_matrix=transformation_matrix,
    )
    transformed_points = transformed_point_cloud.positions

    # Create a color map based on depth values for visualization
    depth_range = np.ptp(depth_array)
    depth_normalized = (depth_array - depth_array.min()) / (depth_range + 1e-12)
    point_colors = np.stack(
        [
            255 * depth_normalized,
            120 + 80 * depth_normalized,
            255 * (1.0 - depth_normalized),
        ],
        axis=-1,
    ).reshape(-1, 3).astype(np.uint8)

    # Initialize Rerun
    rr.init("convert_depth_image_to_point_cloud", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Add EyeControls3D with all parameters for camera movement tuning
    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=np.array([1.4, -1.6, 1.2]),
        look_target=np.array([0.0, 0.0, 0.0]),
        eye_up=np.array([0.0, 0.0, 1.0]),
        spin_speed=0.5,
        speed=0.0,
        tracking_entity=None,
    )

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Input Depth Image", origin="input_depth_image"),
            rrb.Spatial3DView(
                name="Output Point Cloud",
                origin="output_point_cloud",
                background=rrb.Background(color=(255, 255, 255)),
                eye_controls=eye_controls,
                line_grid=rrb.LineGrid3D(visible=False),
                spatial_information=rrb.SpatialInformation(
                    target_frame="tf#/", show_axes=False, show_bounding_box=False
                ),
            ),
        )
    ))
    rr.log("input_depth_image", rr.DepthImage(depth_array, meter=1.0))
    rr.log(
        "output_point_cloud",
        rr.Points3D(transformed_points, colors=point_colors),
        rr.CoordinateFrame("tf#/"),
    )


if __name__ == "__main__":
    convert_depth_image_to_point_cloud_example()
