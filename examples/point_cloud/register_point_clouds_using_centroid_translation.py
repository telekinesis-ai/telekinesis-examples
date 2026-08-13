"""
Demonstrates aligning point clouds by matching their centroids (coarse alignment).

This example:
- Downloads two example point clouds (source and target).
- Computes a translation that moves the source cloud's center to the target cloud's center.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def register_point_clouds_using_centroid_translation_example():
    """
    Aligns point clouds by matching their centroids (coarse alignment).

    Computes a translation that moves the source cloud's center to the target cloud's
    center. Fast initial alignment step before fine registration.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_manufacturing_workpieces.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_manufacturing_workpieces_centered.ply"
    source_point_cloud = datatypes.PointCloud.from_url(url=source_point_cloud_url, use_cache=True)
    target_point_cloud = datatypes.PointCloud.from_url(url=target_point_cloud_url, use_cache=True)
    logger.success(f"Loaded source point cloud with {len(source_point_cloud)} points")
    logger.success(f"Loaded target point cloud with {len(target_point_cloud)} points")

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_centroid_translation(
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )
    logger.success(f"Registered point clouds using centroid translation, transformation_matrix: {transformation_matrix.data}")

    # Access transformation_matrix data and properties
    transformation_matrix_data = transformation_matrix.data
    transformation_matrix_shape = transformation_matrix.shape
    transformation_matrix_ndim = transformation_matrix.ndim
    transformation_matrix_dtype = transformation_matrix.dtype
    logger.info(f"Transformation matrix data: {transformation_matrix_data}")
    logger.info(f"Transformation matrix shape: {transformation_matrix_shape}")
    logger.info(f"Transformation matrix ndim: {transformation_matrix_ndim}")
    logger.info(f"Transformation matrix dtype: {transformation_matrix_dtype}")

    # ===================== Visualization  (Optional) ======================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
        modify_inplace=False,
    )

    rr.init("register_point_clouds_using_centroid_translation_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/before_registration/source")
    datatypes.visualize(target_point_cloud, entity_path="/before_registration/target")
    datatypes.visualize(aligned_source_point_cloud, entity_path="/after_registration/source_aligned")
    datatypes.visualize(target_point_cloud, entity_path="/after_registration/target")


if __name__ == "__main__":
    register_point_clouds_using_centroid_translation_example()
