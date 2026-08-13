"""
Demonstrates scaling a point cloud uniformly about a center point.

This example:
- Downloads an example point cloud.
- Multiplies all point coordinates by a scale factor relative to a center.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def scale_point_cloud_example():
    """
    Scales a point cloud uniformly about a center point.

    Multiplies all point coordinates by a scale factor relative to a center.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/relay_2_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    scaled_point_cloud = vitreous.scale_point_cloud(
        point_cloud=point_cloud,
        center_point=[0.0, 0.0, 0.0],
        scale_factor=0.3,
        modify_inplace=False,
    )
    logger.success(f"Scaled point cloud to {len(scaled_point_cloud)} points")

    # Access scaled_point_cloud data and properties
    scaled_point_cloud_positions = scaled_point_cloud.positions
    scaled_point_cloud_normals = scaled_point_cloud.normals
    scaled_point_cloud_colors = scaled_point_cloud.colors
    logger.info(f"Scaled point cloud positions shape: {scaled_point_cloud_positions.shape}")
    logger.info(f"Scaled point cloud has normals: {scaled_point_cloud_normals is not None}")
    logger.info(f"Scaled point cloud has colors: {scaled_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("scale_point_cloud_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(scaled_point_cloud, entity_path="/scaled_point_cloud")


if __name__ == "__main__":
    scale_point_cloud_example()
