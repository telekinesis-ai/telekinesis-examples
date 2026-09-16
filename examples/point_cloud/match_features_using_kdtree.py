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


def match_features(source_features, target_features):
    """Match feature descriptors while tolerating minor SDK naming differences."""
    from telekinesis import vitreous

    kwargs_options = (
        {
            "source_features": source_features,
            "target_features": target_features,
        },
        {
            "source_feature_descriptors": source_features,
            "target_feature_descriptors": target_features,
        },
        {
            "features1": source_features,
            "features2": target_features,
        },
    )

    last_error = None
    for kwargs in kwargs_options:
        try:
            correspondences = vitreous.match_features_using_kdtree(
                **kwargs,
                use_absolute_scale=False,
                use_crosscheck=True,
                use_tuple_test=False,
                tuple_scale=0.95,
                max_correspondences=5000,
            )
            if isinstance(correspondences, (tuple, list)) and len(correspondences) == 2:
                return np.column_stack(correspondences)
            return correspondences
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("Failed to match features using KD-tree.")


def _log_array_summary(name, array):
    logger.success(f"Results: {array}")
    logger.info(f"{name} shape: {getattr(array, 'shape', None)}")
    logger.info(f"{name} ndim: {getattr(array, 'ndim', None)}")
    logger.info(f"{name} dtype: {getattr(array, 'dtype', None)}")


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
    correspondences = match_features(source_features, target_features)

    # ===================== Log ================================================
    logger.success(
        "Matched source and target point-cloud features using KD-tree search"
    )
    _log_array_summary("Source FPFH feature matrix", source_features)
    _log_array_summary("Target FPFH feature matrix", target_features)
    _log_array_summary("Correspondence set", correspondences)

    # ===================== Visualization  (Optional) ===========================
    rr.init("match_features_using_kdtree_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-source_point_cloud")
    datatypes.visualize(target_point_cloud, entity_path="/2-target_point_cloud")


if __name__ == "__main__":
    match_features_using_kdtree_example()
