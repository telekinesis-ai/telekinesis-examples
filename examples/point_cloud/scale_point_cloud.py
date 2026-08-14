"""
Demonstrates scaling a point cloud uniformly about a center point.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def scale_point_cloud_example():
    """
    Scales a point cloud uniformly about a center point.

    Multiplies all point coordinates by a scale factor relative to a center.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/relay_2_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    scaled_point_cloud = vitreous.scale_point_cloud(
        point_cloud=point_cloud,
        center_point=[0.0, 0.0, 0.0],
        scale_factor=0.3,
        modify_inplace=False,
    )

    # ===================== Log ================================================
    logger.success(f"Scaled point cloud to {len(scaled_point_cloud)} points")
    logger.success(f"Results: {scaled_point_cloud}")
    logger.info(f"Scaled point cloud positions shape: {scaled_point_cloud.positions.shape}")
    logger.info(f"Scaled point cloud has normals shape: "
                f"{scaled_point_cloud.normals.shape if scaled_point_cloud.has_normals else None}")
    logger.info(f"Scaled point cloud has colors shape: "
                f"{scaled_point_cloud.colors.shape if scaled_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("scale_point_cloud_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(scaled_point_cloud, entity_path="/2-scaled_point_cloud")


if __name__ == "__main__":
    scale_point_cloud_example()
