"""
Demonstrates filtering points near a plane defined by a point and normal vector.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_plane_defined_by_point_normal_proximity_example():
    """
    Filters points near a plane defined by a point and normal vector.

    Keeps points within a distance threshold of a plane specified by a point
    on the plane and its normal vector.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_plane_defined_by_point_normal_proximity(
        distance_threshold=4.0,
        point_cloud=point_cloud,
        plane_point=[-15.74520074, 319.25105712, 454.3114797],
        plane_normal=[0.028344755192329624, -0.5747207168510667, -0.8178585895344518],
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using plane defined by point and normal")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}")
    logger.info(f"Filtered point cloud has normals shape: "
                f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}")
    logger.info(f"Filtered point cloud has colors shape: "
                f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_plane_defined_by_point_normal_proximity_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/2-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_plane_defined_by_point_normal_proximity_example()
