"""
Demonstrates computing the geometric center (centroid) of a point cloud.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def calculate_point_cloud_centroid_example():
    """
    Computes the geometric center (centroid) of a point cloud.

    Calculates the mean position of all points in the cloud.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_large_pcb_inspection_cropped_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    # `calculate_point_cloud_centroid` returns a Point3D datatype instance.
    centroid = vitreous.calculate_point_cloud_centroid(point_cloud=point_cloud)

    # ===================== Log ================================================
    logger.success(f"Calculated centroid for {point_cloud}")
    logger.success(f"Results: {centroid}")
    logger.info(f"Centroid data: {centroid.data}")
    logger.info(f"Centroid shape: {centroid.shape}")
    logger.info(f"Centroid dtype: {centroid.dtype}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("calculate_point_cloud_centroid_example", spawn=True)
    datatypes.visualize(point_cloud, centroid, entity_path="/point_cloud")


if __name__ == "__main__":
    calculate_point_cloud_centroid_example()
