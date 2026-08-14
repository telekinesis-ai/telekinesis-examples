"""
Demonstrates clustering a point cloud using the DBSCAN density-based clustering algorithm.
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

    # ===================== Run Skill ==========================================
    clusters = vitreous.cluster_point_cloud_using_dbscan(
        point_cloud=point_cloud,
        max_distance=20,
        min_points=50,
    )

    # ===================== Log ================================================
    logger.success(f"Clustered {point_cloud} using DBSCAN")
    logger.success(f"Results: {clusters}")
    logger.info(f"Number of clusters: {len(clusters)}")
    logger.info(f"Points per cluster: {[len(p) for p in clusters.positions]}")
    logger.info(f"First cluster is a PointCloud with {len(clusters[0])} points")

    # ===================== Visualization  (Optional) ===========================
    rr.init("cluster_point_cloud_using_dbscan_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(clusters, entity_path="/2-dbscan_clusters")


if __name__ == "__main__":
    cluster_point_cloud_using_dbscan_example()
