"""
Demonstrates segmenting points near a line defined by a point and direction vector.
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

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.segment_point_cloud_using_vector_proximity(
        distance_threshold=0.1,
        keep_outliers=False,
        point_cloud=point_cloud,
        reference_point=[0.0, 0.0, 0.0],
        reference_vector=[0.0, 0.0, 1.0],
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {point_cloud} using vector proximity.")
    logger.success(f"Results: {result_point_cloud}")
    logger.info(f"Result point cloud positions shape: {result_point_cloud.positions.shape}")
    logger.info(f"Result point cloud has normals shape: "
                f"{result_point_cloud.normals.shape if result_point_cloud.has_normals else None}")
    logger.info(f"Result point cloud has colors shape: "
                f"{result_point_cloud.colors.shape if result_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("segment_point_cloud_using_vector_proximity_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(result_point_cloud, entity_path="/2-segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_vector_proximity_example()
