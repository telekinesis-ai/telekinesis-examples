"""
Demonstrates splitting a point cloud into regions based on density discontinuities.

This example:
- Downloads an example point cloud.
- Detects and splits point clouds at locations where point density changes dramatically.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def cluster_point_cloud_based_on_density_jump_example():
    """
    Splits a point cloud into regions based on density discontinuities.

    Detects and splits point clouds at locations where point density changes
    dramatically.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/mug_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    clusters = vitreous.cluster_point_cloud_based_on_density_jump(
        point_cloud=point_cloud,
        num_nearest_neighbors=5,
        neighborhood_radius=0.05,
        is_point_cloud_linear=False,
        projection_axis=[0.0, 0.0, 1.0],
    )
    logger.success(f"Split point cloud with {len(point_cloud)} points into {len(clusters)} density-based clusters")

    # ===================== Visualization  (Optional) ======================
    rr.init("cluster_point_cloud_based_on_density_jump_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(clusters, entity_path="/density_jump_clusters")


if __name__ == "__main__":
    cluster_point_cloud_based_on_density_jump_example()
