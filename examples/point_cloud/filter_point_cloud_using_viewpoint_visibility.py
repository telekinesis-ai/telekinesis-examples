"""
Demonstrates filtering points based on visibility from a camera viewpoint.

This example:
- Downloads an example point cloud.
- Removes points that are occluded or outside the visibility range from a specified camera position.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_viewpoint_visibility_example():
    """
    Filters points based on visibility from a camera viewpoint.

    Removes points that are occluded or outside the visibility range from
    a specified camera position.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_parcels_04_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_viewpoint_visibility(
        viewpoint=[100, -500, 250.0],
        visibility_radius=100000.0,
        point_cloud=point_cloud,
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using viewpoint visibility")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_viewpoint_visibility_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_viewpoint_visibility_example()
