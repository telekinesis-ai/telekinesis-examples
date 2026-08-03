"""
Demonstrates reconstructing occupied and free OctoMap cells from a point cloud.

This example downloads a point cloud, builds an OctoMap, and visualizes the
input cloud alongside its occupied and free cells using Rerun.
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


def reconstruct_octomap_example():
    """Reconstructs occupied and free OctoMap cells from a point cloud."""
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/beer_can_corrupted_normals.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    occupied_cells, free_cells = vitreous.reconstruct_octomap(
        point_cloud=point_cloud,
        resolution=0.01,
        sensor_origin=[0.0, 0.0, 0.0],
    )
    logger.success(
        f"Reconstructed OctoMap with {len(occupied_cells.centers)} occupied "
        f"and {len(free_cells.centers)} free cells"
    )

    # ===================== Visualization (Optional) ===========================
    visualize(point_cloud, occupied_cells, free_cells)


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


def visualize(point_cloud, occupied_cells, free_cells) -> None:
    """Visualizes the input point cloud and reconstructed OctoMap using Rerun."""
    # Initialize Rerun
    rr.init("reconstruct_octomap", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Add EyeControls3D with all parameters for camera movement tuning
    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=np.array([0.13403, -0.09305, -0.10270]),
        look_target=np.array([0.00037, -0.00071, 0.06175]),
        eye_up=np.array([0.04087094, 0.0086678, -0.99912684]),
        spin_speed=0.5,
        speed=0.0,
        tracking_entity=None,
    )
    background = rrb.Background(color=(255, 255, 255))
    line_grid = rrb.LineGrid3D(visible=False)

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Input Point Cloud", origin="input_point_cloud",
                background=background, eye_controls=eye_controls, line_grid=line_grid,
            ),
            rrb.Spatial3DView(
                name="Output OctoMap", origin="output_octomap",
                background=background, eye_controls=eye_controls, line_grid=line_grid,
            ),
        )
    ))

    rr.log(
        "input_point_cloud",
        rr.Points3D(
            np.asarray(point_cloud.positions),
            colors=(
                (np.asarray(point_cloud.colors) * 255).astype(np.uint8)
                if point_cloud.has_colors() else None
            ),
        ),
    )
    rr.log(
        "output_octomap/occupied_cells",
        rr.Boxes3D(
            centers=occupied_cells.centers,
            half_sizes=occupied_cells.half_sizes,
            colors=[(0, 120, 255)],
        ),
    )
    rr.log(
        "output_octomap/free_cells",
        rr.Boxes3D(
            centers=free_cells.centers,
            half_sizes=free_cells.half_sizes,
            colors=[(180, 180, 180)],
        ),
    )


if __name__ == "__main__":
    reconstruct_octomap_example()
