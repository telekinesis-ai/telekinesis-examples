"""
Demonstrates computing the oriented bounding box (OBB) of a point cloud.
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
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    oriented_bounding_box = vitreous.calculate_oriented_bounding_box(
        point_cloud=point_cloud,
        minimize_bbox_volume=True,
        use_robust_fitting=True,
    )

    # ===================== Log =================================================
    logger.success(f"Calculated oriented bounding box for {point_cloud}")
    logger.success(f"Results: {oriented_bounding_box}")
    logger.info(f"Oriented bounding box data: {oriented_bounding_box.data}")
    logger.info(f"Oriented bounding box shape: {oriented_bounding_box.shape}")
    logger.info(f"Oriented bounding box center: {oriented_bounding_box.center}")
    logger.info(f"Oriented bounding box size (height, width, depth): "
                f"{oriented_bounding_box.height}, "
                f"{oriented_bounding_box.width}, "
                f"{oriented_bounding_box.depth}")
    logger.info(f"Oriented bounding box volume: {oriented_bounding_box.volume}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("calculate_oriented_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-point_cloud")
    datatypes.visualize(oriented_bounding_box, entity_path="/2-oriented_bounding_box")


if __name__ == "__main__":
    calculate_oriented_bounding_box_example()
