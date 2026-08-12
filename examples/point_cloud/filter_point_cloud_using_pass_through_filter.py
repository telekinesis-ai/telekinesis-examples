"""
Demonstrates filtering points within axis-aligned min/max ranges.

This example:
- Downloads an example point cloud.
- Keeps only points where each coordinate (x, y, z) falls within specified min/max bounds.
- Visualizes the result using Rerun.
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
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

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
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using axis-aligned range")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_pass_through_filter_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_pass_through_filter_example()
