"""
Demonstrates removing points from one cloud that are near points in another cloud.
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

    # ===================== Run Skill ==========================================
    subtracted_point_cloud = vitreous.subtract_point_clouds(
        distance_threshold=0.1,
        point_cloud1=point_cloud1,
        point_cloud2=point_cloud2,
    )

    # ===================== Log ================================================
    logger.success(f"Subtracted {point_cloud2} from {point_cloud1} using distance threshold 0.1")
    logger.success(f"Results: {subtracted_point_cloud}")
    logger.info(f"Subtracted point cloud positions shape: {subtracted_point_cloud.positions.shape}")
    logger.info(f"Subtracted point cloud has normals shape: "
                f"{subtracted_point_cloud.normals.shape if subtracted_point_cloud.has_normals else None}")
    logger.info(f"Subtracted point cloud has colors shape: "
                f"{subtracted_point_cloud.colors.shape if subtracted_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("subtract_point_clouds_example", spawn=True)
    datatypes.visualize(point_cloud1, entity_path="/1-point_cloud_1")
    datatypes.visualize(point_cloud2, entity_path="/2-point_cloud_2")
    datatypes.visualize(subtracted_point_cloud, entity_path="/3-subtracted_point_cloud")


if __name__ == "__main__":
    subtract_point_clouds_example()
