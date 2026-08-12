"""
Demonstrates aligning point clouds by matching their centroids (coarse alignment).

This example:
- Downloads two example point clouds (source and target).
- Computes a translation that moves the source cloud's center to the target cloud's center.
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


def register_point_clouds_using_centroid_translation_example():
    """
    Aligns point clouds by matching their centroids (coarse alignment).

    Computes a translation that moves the source cloud's center to the target cloud's
    center. Fast initial alignment step before fine registration.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_manufacturing_workpieces.ply"
    target_point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/zivid_manufacturing_workpieces_centered.ply"
    source_point_cloud = fetch_point_cloud(source_point_cloud_url)
    target_point_cloud = fetch_point_cloud(target_point_cloud_url)
    logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_centroid_translation(
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )
    logger.success(f"Registered point clouds using centroid translation, transformation_matrix: {transformation_matrix.matrix}")

    # ===================== Visualization  (Optional) ======================
    visualize(source_point_cloud, target_point_cloud)


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


def visualize(source_point_cloud, target_point_cloud) -> None:
    """Visualizes the source and target point clouds using Rerun."""
    # Initialize Rerun
    rr.init("register_point_clouds_using_centroid_translation", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Source Point Cloud", origin="source_point_cloud"),
            rrb.Spatial3DView(name="Target Point Cloud", origin="target_point_cloud")
        ))
    )
    # Log input point clouds
    rr.log(
        "source_point_cloud",
        rr.Points3D(
            positions=source_point_cloud.positions,
            colors=source_point_cloud.colors
        ),
    )
    rr.log(
        "target_point_cloud",
        rr.Points3D(
            positions=target_point_cloud.positions,
            colors=target_point_cloud.colors
        ),
    )


if __name__ == "__main__":
    register_point_clouds_using_centroid_translation_example()
