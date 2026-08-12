"""
Demonstrates removing statistical outliers based on distance distribution.

This example:
- Downloads an example point cloud.
- Removes points that are farther than a threshold from their neighbors, where the threshold is computed from mean distance and standard deviation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_statistical_outlier_removal_example():
    """
    Removes statistical outliers based on distance distribution.

    Removes points that are farther than a threshold from their neighbors,
    where the threshold is computed from mean distance and standard deviation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_6_masked.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_statistical_outlier_removal(
        num_neighbors=90,
        standard_deviation_ratio=0.1,
        point_cloud=point_cloud,
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using statistical outlier removal")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_statistical_outlier_removal_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_statistical_outlier_removal_example()
