"""
Demonstrates aligning point clouds using Point-to-Point Iterative Closest Point (ICP).
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_point_to_point_icp_example():
    """
    Aligns point clouds using Point-to-Point Iterative Closest Point (ICP).

    Iteratively refines alignment by minimizing point-to-point distances.
    Requires good initial alignment.
    """
    # ===================== Load Data ==========================================
    source_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_0_icp_alignment.ply"
    target_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_0_preprocessed.ply"
    source_point_cloud = datatypes.PointCloud.from_url(url=source_url, use_cache=True)
    target_point_cloud = datatypes.PointCloud.from_url(url=target_url, use_cache=True)

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_point_to_point_icp(
        max_iterations=500,
        max_correspondence_distance=10,
        estimate_scaling=False,
        min_fitness_score=0.0001,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using point-to-point ICP"
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
        modify_inplace=False,
    )

    rr.init("register_point_clouds_using_point_to_point_icp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_point_to_point_icp_example()
