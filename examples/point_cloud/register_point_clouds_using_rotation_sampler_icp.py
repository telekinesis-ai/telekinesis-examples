"""
Demonstrates finding the best alignment by trying multiple rotations with ICP refinement.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_rotation_sampler_icp_example():
    """
    Finds best alignment by trying multiple rotations with ICP refinement.

    Samples rotations in Euler angle space, runs ICP for each, and keeps the best.
    """
    # ===================== Load Data ==========================================
    source_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_bottle_cylinder_centered.ply"
    target_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_bottle_segmented.ply"
    source_point_cloud = datatypes.PointCloud.from_url(url=source_url, use_cache=True)
    target_point_cloud = datatypes.PointCloud.from_url(url=target_url, use_cache=True)

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_rotation_sampler_icp(
        x_step_size_deg=30,
        y_step_size_deg=10,
        z_step_size_deg=30,
        x_min_deg=0,
        x_max_deg=180,
        y_min_deg=0,
        y_max_deg=180,
        z_min_deg=0,
        z_max_deg=180,
        early_stop_fitness_score=0.9,
        min_fitness_score=0.2,
        max_iterations=100,
        max_correspondence_distance=2,
        estimate_scaling=False,
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using rotation sampler ICP"
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

    rr.init("register_point_clouds_using_rotation_sampler_icp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_rotation_sampler_icp_example()
