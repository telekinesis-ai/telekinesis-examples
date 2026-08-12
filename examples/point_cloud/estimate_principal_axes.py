"""
Demonstrates computing the principal axes of a point cloud using PCA.

This example:
- Downloads an example point cloud.
- Finds the orthogonal axes along which the point cloud has maximum variance.
- Visualizes the result using Rerun.
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
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    # `estimate_principal_axes` returns a plain (3, 3) numpy array whose columns are the axes.
    principal_axes = vitreous.estimate_principal_axes(
        point_cloud=point_cloud,
        method="obb",
    )
    logger.success("Estimated principal axes")

    # ===================== Visualization  (Optional) ======================
    rr.init("estimate_principal_axes_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/point_cloud")
    datatypes.visualize(datatypes.Vectors3D(principal_axes.T), entity_path="/principal_axes")


if __name__ == "__main__":
    estimate_principal_axes_example()
