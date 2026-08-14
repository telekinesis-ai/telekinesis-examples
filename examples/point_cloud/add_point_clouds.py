"""
Demonstrates merging two point clouds into a single cloud.
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

    # ===================== Run Skill ==========================================
    added_point_cloud = vitreous.add_point_clouds(
        point_cloud1=point_cloud1, 
        point_cloud2=point_cloud2
    )

    # ===================== Log ================================================
    logger.success(f"Added {point_cloud1} and {point_cloud2}")
    logger.success(f"Results: {added_point_cloud}")
    logger.info(f"Added point cloud positions shape: {added_point_cloud.positions.shape}")
    logger.info(f"Added point cloud normals shape: "
                f"{added_point_cloud.normals.shape if added_point_cloud.has_normals else None}")
    logger.info(f"Added point cloud colors shape: "
                f"{added_point_cloud.colors.shape if added_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("add_point_clouds_example", spawn=True)
    datatypes.visualize(point_cloud1, entity_path="/1-point_cloud_1")
    datatypes.visualize(point_cloud2, entity_path="/2-point_cloud_2")
    datatypes.visualize(added_point_cloud, entity_path="/3-added_point_cloud")


if __name__ == "__main__":
    add_point_clouds_example()
