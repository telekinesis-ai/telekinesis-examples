"""
Demonstrates filtering points within an axis-aligned bounding box defined by min/max coordinates.
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

    # ===================== Run Skill ==========================================
    x_min, y_min, z_min, x_max, y_max, z_max = -163, -100, 470, 150, 100, 544
    bbox = datatypes.Boxes3D.from_format(
        [[x_min, y_min, z_min, x_max, y_max, z_max]], source_format="xyzxyz"
    )

    filtered_point_cloud = vitreous.filter_point_cloud_using_bounding_box(
        point_cloud=point_cloud, bbox=bbox
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using bounding box")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}")
    logger.info(f"Filtered point cloud has normals shape: "
                f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}")
    logger.info(f"Filtered point cloud has colors shape: "
                f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_bounding_box_example", spawn=True)
    datatypes.visualize(point_cloud, bbox, entity_path="/1-input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/2-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_bounding_box_example()
