"""
Demonstrates downsampling a point cloud by selecting every Nth point.
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

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_uniform_downsampling(
        step_size=20, point_cloud=point_cloud
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using uniform downsampling")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(f"Filtered point cloud positions shape: {filtered_point_cloud.positions.shape}")
    logger.info(f"Filtered point cloud has normals shape: "
                f"{filtered_point_cloud.normals.shape if filtered_point_cloud.has_normals else None}")
    logger.info(f"Filtered point cloud has colors shape: "
                f"{filtered_point_cloud.colors.shape if filtered_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_uniform_downsampling_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(filtered_point_cloud, entity_path="/2-output_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_uniform_downsampling_example()
