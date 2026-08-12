"""
Demonstrates computing the axis-aligned bounding box (AABB) of a point cloud.

This example:
- Downloads an example point cloud.
- Finds the smallest box aligned with coordinate axes that contains all points.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr
import numpy as np

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
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    axis_aligned_bounding_box = vitreous.calculate_axis_aligned_bounding_box(point_cloud=point_cloud)
    logger.success(f"Calculated axis-aligned bounding box for {len(point_cloud)} points")

    # Access axis_aligned_bounding_box data and properties
    axis_aligned_bounding_box_data = axis_aligned_bounding_box.data
    axis_aligned_bounding_box_shape = axis_aligned_bounding_box.shape
    axis_aligned_bounding_box_center = axis_aligned_bounding_box.center
    axis_aligned_bounding_box_size = (
        axis_aligned_bounding_box.height, 
        axis_aligned_bounding_box.width, 
        axis_aligned_bounding_box.depth
    )
    axis_aligned_bounding_box_volume = axis_aligned_bounding_box.volume

    logger.info(f"Axis-aligned bounding box data: {axis_aligned_bounding_box_data}")
    logger.info(f"Axis-aligned bounding box shape: {axis_aligned_bounding_box_shape}")
    logger.info(f"Axis-aligned bounding box center: {axis_aligned_bounding_box_center}")
    logger.info(f"Axis-aligned bounding box size (height, width, depth): {axis_aligned_bounding_box_size}")
    logger.info(f"Axis-aligned bounding box volume: {axis_aligned_bounding_box_volume}")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_axis_aligned_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, axis_aligned_bounding_box, entity_path="/PointCloud")

if __name__ == "__main__":
    calculate_axis_aligned_bounding_box_example()
