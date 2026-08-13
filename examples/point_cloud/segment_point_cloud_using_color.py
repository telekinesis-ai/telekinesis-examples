"""
Demonstrates segmenting points by color similarity to a target color.

This example:
- Downloads an example point cloud.
- Keeps points whose RGB color is within a distance threshold (Euclidean in RGB space) of a target color.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def segment_point_cloud_using_color_example():
    """
    Segments points by color similarity to a target color.

    Keeps points whose RGB color is within a distance threshold (Euclidean in
    RGB space) of a target color.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    segmented_point_cloud = vitreous.segment_point_cloud_using_color(
        target_color=[50, 75, 200],
        color_distance_threshold=60.0,
        point_cloud=point_cloud,
    )
    logger.success(f"Segmented {len(segmented_point_cloud)} points using color")

    # Access segmented_point_cloud data and properties
    segmented_point_cloud_positions = segmented_point_cloud.positions
    segmented_point_cloud_normals = segmented_point_cloud.normals
    segmented_point_cloud_colors = segmented_point_cloud.colors
    logger.info(f"Segmented point cloud positions shape: {segmented_point_cloud_positions.shape}")
    logger.info(f"Segmented point cloud has normals: {segmented_point_cloud_normals is not None}")
    logger.info(f"Segmented point cloud has colors: {segmented_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_point_cloud_using_color_example", spawn=True)
    # datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(segmented_point_cloud, entity_path="/segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_color_example()
