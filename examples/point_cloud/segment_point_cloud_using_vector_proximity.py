"""
Demonstrates segmenting points near a line defined by a point and direction vector.

This example:
- Downloads an example point cloud.
- Keeps points within a distance threshold of an infinite line through a reference point along a direction.
"""

import pathlib
import tempfile

import requests
from loguru import logger

from datatypes import datatypes, io
from telekinesis import vitreous


def segment_point_cloud_using_vector_proximity_example():
    """
    Segments points near a line defined by a point and direction vector.

    Keeps points within a distance threshold of an infinite line through a
    reference point along a direction.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.segment_point_cloud_using_vector_proximity(
        distance_threshold=0.1,
        keep_outliers=False,
        point_cloud=point_cloud,
        reference_point=[0.0, 0.0, 0.0],
        reference_vector=[0.0, 0.0, 1.0],
    )
    logger.success(
        f"Segmented {len(result_point_cloud.positions)} points using vector proximity"
    )


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
    segment_point_cloud_using_vector_proximity_example()
