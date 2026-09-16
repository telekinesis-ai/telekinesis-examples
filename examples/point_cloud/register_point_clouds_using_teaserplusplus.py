"""
Demonstrates aligning point clouds using TEASER++ with explicit correspondences.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous


def register_point_clouds_using_teaserpp_example():
    """
    Aligns point clouds using TEASER++.

    This example first computes FPFH descriptors for each point cloud, matches
    them with KD-tree search, and then feeds the explicit correspondences into
    TEASER++.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_model_voxelized.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_0_preprocessed_voxelized.ply"
    source_point_cloud = datatypes.PointCloud.from_url(
        url=source_point_cloud_url, use_cache=True
    )
    target_point_cloud = datatypes.PointCloud.from_url(
        url=target_point_cloud_url, use_cache=True
    )
    logger.info(f"Loaded source point cloud: {source_point_cloud}")
    logger.info(f"Loaded target point cloud: {target_point_cloud}")

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_teaserplusplus(
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
        normal_radius=3.7,
        normal_max_neighbors=30,
        feature_radius=11.1,
        feature_max_neighbors=100,
        use_absolute_scale=False,
        use_crosscheck=True,
        use_tuple_test=False,
        tuple_scale=0.95,
        max_correspondences=5000,
        noise_bound=0.01,
        cbar2=1.0,
        estimate_scaling=False,
        rotation_gnc_factor=1.4,
        rotation_max_iterations=10000,
        rotation_cost_threshold=1e-16,
    )

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using TEASER++"
    )
    logger.info(f"Final TEASER++ transformation matrix:\n{transformation_matrix}")
    logger.info(f"Transformation matrix data: {transformation_matrix.data}")
    logger.info(f"Transformation matrix shape: {transformation_matrix.shape}")
    logger.info(f"Transformation matrix ndim: {transformation_matrix.ndim}")
    logger.info(f"Transformation matrix dtype: {transformation_matrix.dtype}")

    # ===================== Visualization  (Optional) ===========================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
    )

    rr.init("register_point_clouds_using_teaserpp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_teaserpp_example()
