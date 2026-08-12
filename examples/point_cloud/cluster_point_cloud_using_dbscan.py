"""
Demonstrates clustering a point cloud using the DBSCAN density-based clustering algorithm.

This example:
- Downloads an example point cloud.
- Identifies clusters of points that are closely packed together, separating distinct objects or regions.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def cluster_point_cloud_using_dbscan_example():
    """
    Clusters a point cloud using the DBSCAN density-based clustering algorithm.

    DBSCAN identifies clusters of points that are closely packed together,
    separating distinct objects or regions.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_bottles_10_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    clusters = vitreous.cluster_point_cloud_using_dbscan(
        point_cloud=point_cloud,
        max_distance=20,
        min_points=50,
    )
    logger.success(f"Clustered point cloud with {len(point_cloud)} points using DBSCAN into {len(clusters)} clusters")

    # ===================== Visualization  (Optional) ======================
    rr.init("cluster_point_cloud_using_dbscan_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(clusters, entity_path="/dbscan_clusters")


if __name__ == "__main__":
    cluster_point_cloud_using_dbscan_example()
