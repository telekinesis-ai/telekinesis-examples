"""
Demonstrates computing the axis-aligned bounding box (AABB) of a point cloud.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def calculate_axis_aligned_bounding_box_example():
    """
    Computes the axis-aligned bounding box (AABB) of a point cloud.

    Finds the smallest box aligned with coordinate axes that contains all points.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_1_raw_preprocessed.ply"
    # By default, the point cloud will be cached in the user cache directory for future runs.
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    axis_aligned_bounding_box = vitreous.calculate_axis_aligned_bounding_box(
        point_cloud=point_cloud
    )

    # ===================== Log =================================================
    logger.success(f"Calculated axis-aligned bounding box for {point_cloud}")
    logger.success(f"Results: {axis_aligned_bounding_box}")
    logger.info(f"Axis-aligned bounding box data: {axis_aligned_bounding_box.data}")
    logger.info(f"Axis-aligned bounding box shape: {axis_aligned_bounding_box.shape}")
    logger.info(f"Axis-aligned bounding box center: {axis_aligned_bounding_box.center}")
    logger.info(
        f"Axis-aligned bounding box dimensions (height, width, depth): "
        f"{axis_aligned_bounding_box.dimensions}, "
    )
    logger.info(f"Axis-aligned bounding box volume: {axis_aligned_bounding_box.volume}")

    # ===================== Visualization  (Optional) ============================
    rr.init("calculate_axis_aligned_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-point_cloud")
    datatypes.visualize(
        axis_aligned_bounding_box, entity_path="/2-axis_aligned_bounding_box"
    )


if __name__ == "__main__":
    calculate_axis_aligned_bounding_box_example()
