"""
Demonstrates filtering points within an oriented (rotated) bounding box.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_oriented_bounding_box_example():
    """
    Filters points within an oriented (rotated) bounding box.

    Keeps only points within a 3D box that can be rotated to any orientation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    x_min, y_min, z_min = -205.65248652, -112.59310319, 554.42936219
    x_max, y_max, z_max = 121.88022318, -17.60647882, 698.54912862
    rot_x, rot_y, rot_z = -38.1245801, -7.89877607, -7.74440359
    bbox = [x_min, y_min, z_min, x_max, y_max, z_max, rot_x, rot_y, rot_z]
    oriented_bbox = datatypes.OrientedBox3D.from_xyzxyz(bbox)

    filtered_point_cloud = vitreous.filter_point_cloud_using_oriented_bounding_box(
        point_cloud=point_cloud, oriented_bbox=oriented_bbox
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using oriented bounding box")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(
        f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}"
    )
    logger.info(
        f"Filtered point cloud has normals shape: "
        f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}"
    )
    logger.info(
        f"Filtered point cloud has colors shape: "
        f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_oriented_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(oriented_bbox, entity_path="/2-oriented_bounding_box")
    datatypes.visualize(filtered_point_cloud, entity_path="/3-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_oriented_bounding_box_example()
