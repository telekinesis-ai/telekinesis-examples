"""
Demonstrates segmenting the dominant plane from a point cloud using RANSAC.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def segment_point_cloud_using_plane_example():
    """
    Segments the dominant plane from a point cloud using RANSAC.

    Finds the largest planar surface in the cloud using random sample consensus.
    Returns inlier points and plane equation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    segmented_point_cloud, plane_model = vitreous.segment_point_cloud_using_plane(
        distance_threshold=1.0,
        num_initial_points=3,
        max_iterations=1000,
        keep_outliers=False,
        point_cloud=point_cloud,
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {len(segmented_point_cloud)} points using plane, plane model: {plane_model.data}")
    logger.success(f"Results: {segmented_point_cloud}, {plane_model}")
    logger.info(f"Segmented point cloud positions shape: {segmented_point_cloud.positions.shape}")
    logger.info(f"Segmented point cloud has normals shape: "
                f"{segmented_point_cloud.normals.shape if segmented_point_cloud.has_normals else None}")
    logger.info(f"Segmented point cloud has colors shape: "
                f"{segmented_point_cloud.colors.shape if segmented_point_cloud.has_colors else None}")
    logger.info(f"Plane model coefficients [a, b, c, d]: {plane_model.data}")
    logger.info(f"Plane model shape: {plane_model.shape}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("segment_point_cloud_using_plane_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(segmented_point_cloud, entity_path="/2-segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_plane_example()
