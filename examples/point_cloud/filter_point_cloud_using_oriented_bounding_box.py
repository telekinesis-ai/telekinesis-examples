"""
Demonstrates filtering points within an oriented (rotated) bounding box.

This example:
- Downloads an example point cloud.
- Keeps only points within a 3D box that can be rotated to any orientation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr
from scipy.spatial.transform import Rotation as R

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_oriented_bounding_box_example():
    """
    Filters points within an oriented (rotated) bounding box.

    Keeps only points within a 3D box that can be rotated to any orientation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    x_min, y_min, z_min = -205.65248652, -112.59310319, 554.42936219
    x_max, y_max, z_max = 121.88022318, -17.60647882, 698.54912862
    rot_x, rot_y, rot_z = -38.1245801, -7.89877607, -7.74440359

    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    size = [x_max - x_min, y_max - y_min, z_max - z_min]
    quat_xyzw = R.from_euler("xyz", [rot_x, rot_y, rot_z], degrees=True).as_quat()
    oriented_bbox = datatypes.OrientedBoxes3D([[*center, *size, *quat_xyzw]])

    filtered_point_cloud = vitreous.filter_point_cloud_using_oriented_bounding_box(
        point_cloud=point_cloud, oriented_bbox=oriented_bbox
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using oriented bounding box")

    # Access filtered_point_cloud data and properties
    filtered_point_cloud_positions = filtered_point_cloud.positions
    filtered_point_cloud_normals = filtered_point_cloud.normals
    filtered_point_cloud_colors = filtered_point_cloud.colors
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud_positions.shape}")
    logger.info(f"Filtered point cloud has normals: {filtered_point_cloud_normals is not None}")
    logger.info(f"Filtered point cloud has colors: {filtered_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_oriented_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, oriented_bbox, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_oriented_bounding_box_example()
