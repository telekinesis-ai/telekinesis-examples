"""
Demonstrates aligning point clouds using Point-to-Plane ICP.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_point_to_plane_icp_example():
    """
    Aligns point clouds using Point-to-Plane ICP.

    Minimizes point-to-tangent-plane distances instead of point-to-point. More
    accurate than point-to-point ICP, especially for planar surfaces.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/weld_clamp_model_shifted.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/weld_clamp_cluster_0_centroid_registered.ply"
    source_point_cloud = datatypes.PointCloud.from_url(
        url=source_point_cloud_url, use_cache=True
    )
    target_point_cloud = datatypes.PointCloud.from_url(
        url=target_point_cloud_url, use_cache=True
    )

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_point_to_plane_icp(
        max_iterations=500,
        max_correspondence_distance=30,
        normal_max_neighbors=20,
        normal_search_radius=2,
        use_robust_kernel=False,
        loss_type="tukey_loss",
        noise_standard_deviation=10,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using point-to-plane ICP"
    )
    logger.success(f"Results: {transformation_matrix}")
    logger.info(f"Transformation matrix data: {transformation_matrix.data}")
    logger.info(f"Transformation matrix shape: {transformation_matrix.shape}")
    logger.info(f"Transformation matrix ndim: {transformation_matrix.ndim}")
    logger.info(f"Transformation matrix dtype: {transformation_matrix.dtype}")

    # ===================== Visualization  (Optional) ===========================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
    )

    rr.init("register_point_clouds_using_point_to_plane_icp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_point_to_plane_icp_example()
