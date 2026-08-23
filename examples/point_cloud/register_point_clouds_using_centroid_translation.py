"""
Demonstrates aligning point clouds by matching their centroids (coarse alignment).
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
    source_point_cloud = datatypes.PointCloud.from_url(
        url=source_point_cloud_url, use_cache=True
    )
    target_point_cloud = datatypes.PointCloud.from_url(
        url=target_point_cloud_url, use_cache=True
    )

    # ===================== Run Skill ==========================================
    transformation_matrix = vitreous.register_point_clouds_using_centroid_translation(
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
    )

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using centroid translation"
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

    rr.init("register_point_clouds_using_centroid_translation_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_centroid_translation_example()
