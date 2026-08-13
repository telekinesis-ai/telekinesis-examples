"""
Demonstrates applying a 6-DOF rigid transformation (rotation + translation) to a point cloud.

This example:
- Downloads an example point cloud.
- Transforms points using a 4x4 homogeneous transformation matrix.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def apply_transform_to_point_cloud_example():
    """
    Applies a 6-DOF rigid transformation (rotation + translation) to a point cloud.

    Transforms points using a 4x4 homogeneous transformation matrix.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/plastic_centered.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    transformed_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=point_cloud,
        transformation_matrix=[[1, 0, 0, 15], [0, 1, 0, 15], [0, 0, 1, 5], [0, 0, 0, 1]],
        modify_inplace=False,
    )
    logger.success(f"Applied transform to {len(transformed_point_cloud)} points")

    # Access transformed_point_cloud data and properties
    transformed_point_cloud_positions = transformed_point_cloud.positions
    transformed_point_cloud_normals = transformed_point_cloud.normals
    transformed_point_cloud_colors = transformed_point_cloud.colors
    logger.info(f"Transformed point cloud positions shape: {transformed_point_cloud_positions.shape}")
    logger.info(f"Transformed point cloud has normals: {transformed_point_cloud_normals is not None}")
    logger.info(f"Transformed point cloud has colors: {transformed_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("apply_transform_to_point_cloud_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/source_point_cloud")
    datatypes.visualize(transformed_point_cloud, entity_path="/transformed_point_cloud")


if __name__ == "__main__":
    apply_transform_to_point_cloud_example()
