"""
Demonstrates aligning point clouds using Fast Global Registration (FGR).

This example:
- Downloads two example point clouds (source and target).
- Runs feature-based Fast Global Registration using graduated non-convexity optimization.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_fast_global_registration_example():
    """
    Aligns point clouds using Fast Global Registration (FGR).

    Feature-based registration that's faster than RANSAC. Uses graduated
    non-convexity optimization.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_model_voxelized.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_0_preprocessed_voxelized.ply"
    source_point_cloud = datatypes.PointCloud.from_url(url=source_point_cloud_url, use_cache=True)
    target_point_cloud = datatypes.PointCloud.from_url(url=target_point_cloud_url, use_cache=True)
    logger.success(f"Loaded source point cloud with {len(source_point_cloud)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud)} points")

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_fast_global_registration(
        normal_radius=3.7,
        normal_max_neighbors=30,
        feature_radius=11.1,
        feature_max_neighbors=100,
        max_correspondence_distance=7.4,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )
    logger.success(f"Registered point clouds using fast global registration, transformation_matrix: {transformation_matrix.data}")

    # ===================== Visualization  (Optional) ======================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
        modify_inplace=False,
    )

    rr.init("register_point_clouds_using_fast_global_registration_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/before_registration/source")
    datatypes.visualize(target_point_cloud, entity_path="/before_registration/target")
    datatypes.visualize(aligned_source_point_cloud, entity_path="/after_registration/source_aligned")
    datatypes.visualize(target_point_cloud, entity_path="/after_registration/target")


if __name__ == "__main__":
    register_point_clouds_using_fast_global_registration_example()
