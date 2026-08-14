"""
Demonstrates downsampling a point cloud using voxel grid averaging.
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

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_voxel_downsampling(
        voxel_size=0.005, point_cloud=point_cloud
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using voxel downsampling")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(
        f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}"
    )
    logger.info(
        f"Filtered point cloud has normals shape: "
        f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}"
    )
    logger.info(
        f"Filtered point cloud has colors shape: "
        f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_voxel_downsampling_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/2-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_voxel_downsampling_example()
