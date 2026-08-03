"""
Demonstrates counting the number of points in a point cloud.

This example:
- Downloads an example point cloud.
- Returns the total point count.
"""

import pathlib
import tempfile

import requests
from loguru import logger

from datatypes import datatypes, io
from telekinesis import vitreous


def calculate_points_in_point_cloud_example():
    """
    Counts the number of points in a point cloud.

    Simple utility that returns the total point count.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_1_raw.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    num_points = vitreous.calculate_points_in_point_cloud(point_cloud=point_cloud)
    logger.success(f"Counted {num_points.value} points in point cloud")


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


if __name__ == "__main__":
    calculate_points_in_point_cloud_example()
