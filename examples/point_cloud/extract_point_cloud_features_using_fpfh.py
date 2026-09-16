"""
Demonstrates extracting FPFH descriptors from a point cloud.
"""

from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous

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
    logger.info(f"Loaded source point cloud: {source_point_cloud}")
    logger.info(f"Loaded target point cloud: {target_point_cloud}")

    # ===================== Run Skill ==========================================
    features = vitreous.extract_point_cloud_features_using_fpfh(
        point_cloud=source_point_cloud,
        normal_radius=0.002,
        normal_max_neighbors=20,
        feature_radius=0.005,
        feature_max_neighbors=30,
    )

    # ===================== Log ================================================
    logger.success(f"Extracted FPFH features for {source_point_cloud}")
    logger.success(f"Results: {features}")
    logger.info(
        f"FPFH feature matrix: shape={features.shape}, ndim={features.ndim}, dtype={features.dtype}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("extract_point_cloud_features_using_fpfh_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-source_point_cloud")
    datatypes.visualize(target_point_cloud, entity_path="/2-target_point_cloud")


if __name__ == "__main__":
    extract_point_cloud_features_using_fpfh_example()
