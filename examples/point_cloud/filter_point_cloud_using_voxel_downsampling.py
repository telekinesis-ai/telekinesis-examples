"""
Demonstrates downsampling a point cloud using voxel grid averaging.

This example:
- Downloads an example point cloud.
- Divides 3D space into voxels and replaces all points within each voxel with their centroid.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_voxel_downsampling_example():
    """
    Downsamples a point cloud using voxel grid averaging.

    Divides 3D space into voxels and replaces all points within each voxel
    with their centroid.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_1_subtracted.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_voxel_downsampling(
        voxel_size=0.005, point_cloud=point_cloud
    )
    logger.success(f"Filtered point cloud to {len(filtered_point_cloud)} points using voxel downsampling")

    # Access filtered_point_cloud data and properties
    filtered_point_cloud_positions = filtered_point_cloud.positions
    filtered_point_cloud_normals = filtered_point_cloud.normals
    filtered_point_cloud_colors = filtered_point_cloud.colors
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud_positions.shape}")
    logger.info(f"Filtered point cloud has normals: {filtered_point_cloud_normals is not None}")
    logger.info(f"Filtered point cloud has colors: {filtered_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_voxel_downsampling_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_voxel_downsampling_example()
