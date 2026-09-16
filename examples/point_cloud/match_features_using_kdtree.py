"""
Demonstrates matching FPFH descriptors using KD-tree nearest-neighbor search.
"""

from loguru import logger
import rerun as rr

from telekinesis import datatypes

import numpy as np


def extract_fpfh_features(point_cloud):
    """Extract FPFH descriptors using spacing-scaled neighborhood parameters."""
    from telekinesis import vitreous

    return vitreous.extract_point_cloud_features_using_fpfh(
        point_cloud=point_cloud,
        normal_radius=0.002,
        normal_max_neighbors=20,
        feature_radius=0.005,
        feature_max_neighbors=30,
    )


def match_features_using_kdtree_example():
    """
    Match source and target FPFH descriptors with a KD-tree.

    The output correspondences can be fed into downstream registration methods
    such as TEASER++.
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
    source_features = extract_fpfh_features(source_point_cloud)
    target_features = extract_fpfh_features(target_point_cloud)

    from telekinesis import vitreous

    correspondences = vitreous.match_features_using_kdtree(
        source_features=source_features,
        target_features=target_features,
        cross_check=True,
        tuple_test=False,
        tuple_scale=0.95,
        max_correspondences=5000,
    )
    if isinstance(correspondences, (tuple, list)) and len(correspondences) == 2:
        correspondences = np.column_stack(correspondences)

    # ===================== Log ================================================
    logger.success(
        "Matched source and target point-cloud features using KD-tree search"
    )
    logger.success(f"Source FPFH features: {source_features}")
    logger.info(
        f"Source FPFH feature matrix: shape={source_features.shape}, ndim={source_features.ndim}, dtype={source_features.dtype}"
    )
    logger.success(f"Target FPFH features: {target_features}")
    logger.info(
        f"Target FPFH feature matrix: shape={target_features.shape}, ndim={target_features.ndim}, dtype={target_features.dtype}"
    )
    logger.success(f"Correspondences: {correspondences}")
    logger.info(
        f"Correspondence set: shape={correspondences.shape}, ndim={correspondences.ndim}, dtype={correspondences.dtype}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("match_features_using_kdtree_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-source_point_cloud")
    datatypes.visualize(target_point_cloud, entity_path="/2-target_point_cloud")


if __name__ == "__main__":
    match_features_using_kdtree_example()
