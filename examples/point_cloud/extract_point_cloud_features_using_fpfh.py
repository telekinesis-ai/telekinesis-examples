"""
Demonstrates extracting FPFH descriptors from a point cloud.
"""

import numpy as np
from loguru import logger
import rerun as rr
from scipy.spatial import cKDTree

from telekinesis import datatypes, vitreous


def estimate_point_spacing(point_cloud):
    """Estimate median nearest-neighbor spacing for a point cloud."""
    points = np.asarray(point_cloud.positions)
    if len(points) < 2:
        return 0.0

    tree = cKDTree(points)

    # k=2 because the closest point is the point itself.
    distances, _ = tree.query(points, k=2)
    nearest_neighbor_distances = distances[:, 1]
    return float(np.median(nearest_neighbor_distances))


def _log_feature_summary(feature_name, features):
    logger.success(f"Results: {features}")
    logger.info(f"{feature_name} shape: {getattr(features, 'shape', None)}")
    logger.info(f"{feature_name} ndim: {getattr(features, 'ndim', None)}")
    logger.info(f"{feature_name} dtype: {getattr(features, 'dtype', None)}")


def extract_point_cloud_features_using_fpfh_example():
    """
    Extract Fast Point Feature Histogram descriptors from a point cloud.

    FPFH encodes local geometry around each point and is commonly used as a
    lightweight descriptor for feature-based registration.
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
    spacing = estimate_point_spacing(source_point_cloud)
    logger.info(f"Loaded source point cloud: {source_point_cloud}")
    logger.info(f"Loaded target point cloud: {target_point_cloud}")
    logger.info(f"Estimated point spacing: {spacing}")

    # ===================== Run Skill ==========================================
    features = vitreous.extract_point_cloud_features_using_fpfh(
        point_cloud=source_point_cloud,
        normal_radius=spacing * 2.0,
        normal_max_neighbors=30,
        feature_radius=spacing * 5.0,
        feature_max_neighbors=100,
    )

    # ===================== Log ================================================
    logger.success(f"Extracted FPFH features for {source_point_cloud}")
    _log_feature_summary("FPFH feature matrix", features)

    # ===================== Visualization  (Optional) ===========================
    rr.init("extract_point_cloud_features_using_fpfh_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-source_point_cloud")
    datatypes.visualize(target_point_cloud, entity_path="/2-target_point_cloud")


if __name__ == "__main__":
    extract_point_cloud_features_using_fpfh_example()
