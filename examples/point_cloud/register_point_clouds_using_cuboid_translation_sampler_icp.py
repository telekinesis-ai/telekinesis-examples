"""
Demonstrates finding the best alignment by sampling translations in a 3D grid (cuboid) with ICP.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_cuboid_translation_sampler_icp_example():
    """
    Finds best alignment by sampling translations in a 3D grid (cuboid) with ICP.

    Tries translations on a regular 3D grid within specified x/y/z ranges, runs ICP
    for each, and keeps best result.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/weld_clamp_model_shifted.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/weld_clamp_cluster_0_centroid_registered.ply"
    source_point_cloud = datatypes.PointCloud.from_url(url=source_point_cloud_url, use_cache=True)
    target_point_cloud = datatypes.PointCloud.from_url(url=target_point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_cuboid_translation_sampler_icp(
        step_size=2,
        x_min=-20,
        x_max=20,
        y_min=-20,
        y_max=20,
        z_min=-20,
        z_max=20,
        early_stop_fitness_score=0.3,
        min_fitness_score=0.48,
        max_iterations=50,
        max_correspondence_distance=2,
        estimate_scaling=False,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )

    # ===================== Log ================================================
    logger.success(f"Registered {source_point_cloud} to {target_point_cloud} using cuboid translation sampler ICP")
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

    rr.init("register_point_clouds_using_cuboid_translation_sampler_icp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned")


if __name__ == "__main__":
    register_point_clouds_using_cuboid_translation_sampler_icp_example()
