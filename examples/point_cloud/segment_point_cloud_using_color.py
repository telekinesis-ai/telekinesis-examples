"""
Demonstrates segmenting points by color similarity to a target color.
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
    point_cloud_url = (
        "https://assets.telekinesis.ai/examples/v1/point_clouds/engine_parts_0.ply"
    )
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    segmented_point_cloud = vitreous.segment_point_cloud_using_color(
        point_cloud=point_cloud,
        target_color=[50, 75, 200],
        color_distance_threshold=60.0,
    )

    # ===================== Log ================================================
    logger.success(
        f"Segmented {point_cloud} by color similarity to target color [50, 75, 200] with distance threshold 60.0"
    )
    logger.success(f"Results: {segmented_point_cloud}")
    logger.info(
        f"Segmented point cloud positions shape: {segmented_point_cloud.positions.shape}"
    )
    logger.info(
        f"Segmented point cloud has normals shape: "
        f"{segmented_point_cloud.normals.shape if segmented_point_cloud.has_normals else None}"
    )
    logger.info(
        f"Segmented point cloud has colors shape: "
        f"{segmented_point_cloud.colors.shape if segmented_point_cloud.has_colors else None}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("segment_point_cloud_using_color_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(segmented_point_cloud, entity_path="/2-segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_color_example()
