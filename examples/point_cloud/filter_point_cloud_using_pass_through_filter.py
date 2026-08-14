"""
Demonstrates filtering points within axis-aligned min/max ranges.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_pass_through_filter_example():
    """
    Filters points within axis-aligned min/max ranges.

    Keeps only points where each coordinate (x, y, z) falls within specified
    min/max bounds.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/mounts_3_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_pass_through_filter(
        x_min=-185.0,
        x_max=230.0,
        y_min=-164.0,
        y_max=164.0,
        z_min=450.0,
        z_max=548.0,
        point_cloud=point_cloud,
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using axis-aligned range")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}")
    logger.info(f"Filtered point cloud has normals shape: "
                f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}")
    logger.info(f"Filtered point cloud has colors shape: "
                f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_pass_through_filter_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/2-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_pass_through_filter_example()
