"""
Demonstrates filtering points within an axis-aligned bounding box.

This example:
- Downloads an example point cloud.
- Keeps only points that fall within the specified 3D box defined by min/max coordinates.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_bounding_box_example():
    """
    Filters points within an axis-aligned bounding box.

    Keeps only points that fall within the specified 3D box defined by
    min/max coordinates along each axis.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/plastic_2_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    x_min, y_min, z_min, x_max, y_max, z_max = -163, -100, 470, 150, 100, 544
    bbox = datatypes.Boxes3D.from_format(
        [[x_min, y_min, z_min, x_max, y_max, z_max]], source_format="xyzxyz"
    )

    filtered_point_cloud = vitreous.filter_point_cloud_using_bounding_box(
        point_cloud=point_cloud, bbox=bbox
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using bounding box")

    # Access filtered_point_cloud data and properties
    filtered_point_cloud_positions = filtered_point_cloud.positions
    filtered_point_cloud_normals = filtered_point_cloud.normals
    filtered_point_cloud_colors = filtered_point_cloud.colors
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud_positions.shape}")
    logger.info(f"Filtered point cloud has normals: {filtered_point_cloud_normals is not None}")
    logger.info(f"Filtered point cloud has colors: {filtered_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, bbox, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_bounding_box_example()
