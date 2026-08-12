"""
Demonstrates segmenting the dominant plane from a point cloud using RANSAC.

This example:
- Downloads an example point cloud.
- Finds the largest planar surface in the cloud using random sample consensus and returns inlier points and the plane equation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def segment_point_cloud_using_plane_example():
    """
    Segments the dominant plane from a point cloud using RANSAC.

    Finds the largest planar surface in the cloud using random sample consensus.
    Returns inlier points and plane equation.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_3_downsampled.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    segmented_point_cloud, plane_model = vitreous.segment_point_cloud_using_plane(
        distance_threshold=1.0,
        num_initial_points=3,
        max_iterations=1000,
        keep_outliers=False,
        point_cloud=point_cloud,
    )
    logger.success(f"Segmented {len(segmented_point_cloud)} points using plane, plane model: {plane_model.data}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_point_cloud_using_plane_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(segmented_point_cloud, entity_path="/segmented_point_cloud")


if __name__ == "__main__":
    segment_point_cloud_using_plane_example()
