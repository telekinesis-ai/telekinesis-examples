"""
Demonstrates computing the principal axes of a point cloud using PCA.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def estimate_principal_axes_example():
    """
    Computes the principal axes of a point cloud using PCA.

    Finds the orthogonal axes along which the point cloud has maximum variance.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_large_pcb_inspection_cropped_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    principal_axes = vitreous.estimate_principal_axes(
        point_cloud=point_cloud,
        method="obb",
    )

    # ===================== Log ================================================
    logger.success(f"Estimated principal axes for {point_cloud}")
    logger.success(f"Results: {principal_axes}")
    logger.info(f"Principal axes data (columns are the principal axis vectors): {principal_axes.data}")
    logger.info(f"Principal axes shape: {principal_axes.shape}")
    logger.info(f"Principal axes dtype: {principal_axes.dtype}")
    logger.info(f"Principal axes columns are orthonormal: {principal_axes.is_orthonormal()}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("estimate_principal_axes_example", spawn=True)
    datatypes.visualize(point_cloud, principal_axes, entity_path="/point_cloud")


if __name__ == "__main__":
    estimate_principal_axes_example()
