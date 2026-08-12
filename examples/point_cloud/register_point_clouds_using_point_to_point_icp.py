"""
Demonstrates aligning point clouds using Point-to-Point Iterative Closest Point (ICP).

This example:
- Downloads two example point clouds (source and target).
- Iteratively refines alignment by minimizing point-to-point distances.
- Visualizes the result using Rerun.
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
    logger.success(f"Loaded source point cloud with {len(source_point_cloud)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud)} points")

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
    logger.success(f"Registered point clouds using point-to-point ICP, transformation_matrix: {transformation_matrix.data}")

    # ===================== Visualization  (Optional) ======================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
        modify_inplace=False,
    )

    rr.init("register_point_clouds_using_point_to_point_icp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/before_registration/source")
    datatypes.visualize(target_point_cloud, entity_path="/before_registration/target")
    datatypes.visualize(aligned_source_point_cloud, entity_path="/after_registration/source_aligned")
    datatypes.visualize(target_point_cloud, entity_path="/after_registration/target")


if __name__ == "__main__":
    register_point_clouds_using_point_to_point_icp_example()
