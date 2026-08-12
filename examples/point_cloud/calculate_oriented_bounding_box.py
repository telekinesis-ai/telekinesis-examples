"""
Demonstrates computing the oriented bounding box (OBB) of a point cloud.

This example:
- Downloads an example point cloud.
- Finds the smallest box (in any orientation) that contains all points.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def calculate_oriented_bounding_box_example():
    """
    Computes the oriented bounding box (OBB) of a point cloud.

    Finds the smallest box (in any orientation) that contains all points.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_1_raw_obb_preprocessed.ply"
    # By default, the point cloud will be cached in the user cache directory for future runs.
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    oriented_bounding_box = vitreous.calculate_oriented_bounding_box(
        point_cloud=point_cloud,
        minimize_bbox_volume=True,
        use_robust_fitting=True,
    )
    logger.success(f"Calculated oriented bounding box for {len(point_cloud)} points")

    # Access oriented_bounding_box data and properties
    oriented_bounding_box_data = oriented_bounding_box.data
    oriented_bounding_box_shape = oriented_bounding_box.shape
    oriented_bounding_box_center = oriented_bounding_box.center
    oriented_bounding_box_size = (
        oriented_bounding_box.height,
        oriented_bounding_box.width,
        oriented_bounding_box.depth,
    )
    oriented_bounding_box_volume = oriented_bounding_box.volume

    logger.info(f"Oriented bounding box data: {oriented_bounding_box_data}")
    logger.info(f"Oriented bounding box shape: {oriented_bounding_box_shape}")
    logger.info(f"Oriented bounding box center: {oriented_bounding_box_center}")
    logger.info(f"Oriented bounding box size (height, width, depth): {oriented_bounding_box_size}")
    logger.info(f"Oriented bounding box volume: {oriented_bounding_box_volume}")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_oriented_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, oriented_bounding_box, entity_path="/PointCloud")


if __name__ == "__main__":
    calculate_oriented_bounding_box_example()
