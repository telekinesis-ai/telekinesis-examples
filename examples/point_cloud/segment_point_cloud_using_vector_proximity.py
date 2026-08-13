"""
Demonstrates segmenting points near a line defined by a point and direction vector.

This example:
- Downloads an example point cloud.
- Keeps points within a distance threshold of an infinite line through a reference point along a direction.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def segment_point_cloud_using_vector_proximity_example():
    """
    Segments points near a line defined by a point and direction vector.

    Keeps points within a distance threshold of an infinite line through a
    reference point along a direction.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.segment_point_cloud_using_vector_proximity(
        distance_threshold=0.1,
        keep_outliers=False,
        point_cloud=point_cloud,
        reference_point=[0.0, 0.0, 0.0],
        reference_vector=[0.0, 0.0, 1.0],
    )
    logger.success(f"Segmented {len(result_point_cloud)} points using vector proximity")

    # Access result_point_cloud data and properties
    result_point_cloud_positions = result_point_cloud.positions
    result_point_cloud_normals = result_point_cloud.normals
    result_point_cloud_colors = result_point_cloud.colors
    logger.info(f"Result point cloud positions shape: {result_point_cloud_positions.shape}")
    logger.info(f"Result point cloud has normals: {result_point_cloud_normals is not None}")
    logger.info(f"Result point cloud has colors: {result_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_point_cloud_using_vector_proximity_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(result_point_cloud, entity_path="/segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_vector_proximity_example()
