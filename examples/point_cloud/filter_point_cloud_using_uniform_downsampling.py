"""
Demonstrates downsampling a point cloud by selecting every Nth point.

This example:
- Downloads an example point cloud.
- Uniformly samples points by selecting every step_size-th point from the original cloud.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_uniform_downsampling_example():
    """
    Downsamples a point cloud by selecting every Nth point.

    Uniformly samples points by selecting every step_size-th point from the
    original cloud.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_welding_scene.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_uniform_downsampling(
        step_size=20, point_cloud=point_cloud
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using uniform downsampling")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_uniform_downsampling_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_uniform_downsampling_example()
