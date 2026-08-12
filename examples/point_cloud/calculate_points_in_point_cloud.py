"""
Demonstrates counting the number of points in a point cloud.

This example:
- Downloads an example point cloud.
- Returns the total point count.
"""

from loguru import logger

from telekinesis import vitreous, datatypes


def calculate_points_in_point_cloud_example():
    """
    Counts the number of points in a point cloud.

    Simple utility that returns the total point count.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_1_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    num_points = vitreous.calculate_points_in_point_cloud(point_cloud=point_cloud)
    logger.success(f"Counted {num_points.data} points in point cloud")


if __name__ == "__main__":
    calculate_points_in_point_cloud_example()
