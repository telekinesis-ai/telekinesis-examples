"""
Demonstrates filtering points near a plane defined by coefficients.

This example:
- Downloads an example point cloud.
- Keeps points within a distance threshold of a plane specified by its equation coefficients (ax + by + cz + d = 0).
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_plane_proximity_example():
    """
    Filters points near a plane defined by coefficients.

    Keeps points within a distance threshold of a plane specified by its
    equation coefficients (ax + by + cz + d = 0).
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_plane_proximity(
        distance_threshold=4.0,
        point_cloud=point_cloud,
        plane_coefficients=[0.028344755192329624, -0.5747207168510667, -0.8178585895344518, 555.4890362620131],
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using plane proximity")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_plane_proximity_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_plane_proximity_example()
