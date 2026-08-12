"""
Demonstrates splitting a point cloud by a plane, keeping one side.

This example:
- Downloads an example point cloud.
- Divides a point cloud using a plane and keeps points on either the positive or negative side.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_plane_splitting_example():
    """
    Splits a point cloud by a plane, keeping one side.

    Divides a point cloud using a plane and keeps points on either the positive
    or negative side.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/mounts_3_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_plane_splitting(
        keep_positive_side=False,
        point_cloud=point_cloud,
        plane_coefficients=[0, 0, 1, -547],
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using plane splitting")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_plane_splitting_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_plane_splitting_example()
