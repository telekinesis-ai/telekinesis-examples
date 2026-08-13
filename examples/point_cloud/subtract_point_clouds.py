"""
Demonstrates removing points from one cloud that are near points in another cloud.

This example:
- Downloads two example point clouds.
- Subtracts point_cloud2 from point_cloud1 by removing any point in cloud1 within a distance threshold of any point in cloud2.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def subtract_point_clouds_example():
    """
    Removes points from one cloud that are near points in another cloud.

    Subtracts point_cloud2 from point_cloud1 by removing any point in cloud1
    that is within distance_threshold of any point in cloud2.
    """
    # ===================== Load Data ==========================================
    point_cloud_url_1 = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_mixed_grocery_pallet_centered.ply"
    point_cloud_url_2 = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_mixed_grocery_pallet_box_filtered.ply"
    point_cloud1 = datatypes.PointCloud.from_url(url=point_cloud_url_1, use_cache=True)
    point_cloud2 = datatypes.PointCloud.from_url(url=point_cloud_url_2, use_cache=True)
    logger.success(f"Loaded point cloud 1 with {len(point_cloud1)} points")
    logger.success(f"Loaded point cloud 2 with {len(point_cloud2)} points")

    # ===================== Run Skill ==========================================
    subtracted_point_cloud = vitreous.subtract_point_clouds(
        distance_threshold=0.1,
        point_cloud1=point_cloud1,
        point_cloud2=point_cloud2,
    )
    logger.success(f"Subtracted point clouds to {len(subtracted_point_cloud)} points")

    # Access subtracted_point_cloud data and properties
    subtracted_point_cloud_positions = subtracted_point_cloud.positions
    subtracted_point_cloud_normals = subtracted_point_cloud.normals
    subtracted_point_cloud_colors = subtracted_point_cloud.colors
    logger.info(f"Subtracted point cloud positions shape: {subtracted_point_cloud_positions.shape}")
    logger.info(f"Subtracted point cloud has normals: {subtracted_point_cloud_normals is not None}")
    logger.info(f"Subtracted point cloud has colors: {subtracted_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("subtract_point_clouds_example", spawn=True)
    datatypes.visualize(point_cloud1, entity_path="/point_cloud_1")
    datatypes.visualize(point_cloud2, entity_path="/point_cloud_2")
    datatypes.visualize(subtracted_point_cloud, entity_path="/subtracted_point_cloud")


if __name__ == "__main__":
    subtract_point_clouds_example()
