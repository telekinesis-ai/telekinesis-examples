"""
Demonstrates computing the geometric center (centroid) of a point cloud.

This example:
- Downloads an example point cloud.
- Calculates the mean position of all points in the cloud.
- Visualizes the result using Rerun.
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
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    # `calculate_point_cloud_centroid` returns a plain numpy array, not a datatype instance.
    centroid = vitreous.calculate_point_cloud_centroid(point_cloud=point_cloud)
    logger.success(f"Calculated centroid {centroid} for {len(point_cloud)} points")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_point_cloud_centroid_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/point_cloud")
    datatypes.visualize(datatypes.Position3D(centroid), entity_path="/centroid", label="Centroid")


if __name__ == "__main__":
    calculate_point_cloud_centroid_example()
