"""
Demonstrates merging two point clouds into a single cloud.

This example:
- Downloads two example point clouds.
- Combines all points from both clouds into one unified point cloud.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def add_point_clouds_example():
    """
    Merges two point clouds into a single cloud.

    Combines all points from both clouds into one unified point cloud.
    """
    # ===================== Load Data ==========================================
    point_cloud_url_1 = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_clustered.ply"
    point_cloud_url_2 = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_segmented_plane.ply"
    point_cloud1 = datatypes.PointCloud.from_url(url=point_cloud_url_1, use_cache=True)
    point_cloud2 = datatypes.PointCloud.from_url(url=point_cloud_url_2, use_cache=True)
    logger.success(f"Loaded point cloud 1 with {len(point_cloud1)} points")
    logger.success(f"Loaded point cloud 2 with {len(point_cloud2)} points")

    # ===================== Run Skill ==========================================
    added_point_cloud = vitreous.add_point_clouds(
        point_cloud1=point_cloud1, point_cloud2=point_cloud2
    )
    logger.success(f"Added point clouds: {len(point_cloud1)} + {len(point_cloud2)} points")

    # Access added_point_cloud data and properties
    added_point_cloud_positions = added_point_cloud.positions
    added_point_cloud_normals = added_point_cloud.normals
    added_point_cloud_colors = added_point_cloud.colors
    logger.info(f"Added point cloud positions shape: {added_point_cloud_positions.shape}")
    logger.info(f"Added point cloud has normals: {added_point_cloud_normals is not None}")
    logger.info(f"Added point cloud has colors: {added_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("add_point_clouds_example", spawn=True)
    datatypes.visualize(point_cloud1, entity_path="/point_cloud_1")
    datatypes.visualize(point_cloud2, entity_path="/point_cloud_2")
    datatypes.visualize(added_point_cloud, entity_path="/added_point_cloud")


if __name__ == "__main__":
    add_point_clouds_example()
