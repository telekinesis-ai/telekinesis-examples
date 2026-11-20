"""
Demonstrates applying a 6-DOF rigid transformation (rotation + translation) to a point cloud.
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
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    point_cloud_url = (
        "https://assets.telekinesis.ai/examples/v1/point_clouds/xyzibd_gears_bin_camera_frame.ply"
    )
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    transformed_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=point_cloud,
        transformation_matrix=[
            [0.9995, -0.0028, 0.0316, 4.6],
            [0.0, -0.9962, -0.0870, 66.4],
            [0.0317, 0.0870, -0.9957, 825.3],
            [0, 0, 0, 1],
        ]
    )

    # ===================== Log ================================================
    logger.success(f"Applied transform to {point_cloud}")
    logger.success(f"Results: {transformed_point_cloud}")
    logger.info(
        f"Transformed point cloud positions shape: {transformed_point_cloud.positions.shape}"
    )
    logger.info(
        f"Transformed point cloud has normals shape: "
        f"{transformed_point_cloud.normals.shape if transformed_point_cloud.has_normals else None}"
    )
    logger.info(
        f"Transformed point cloud has colors shape: "
        f"{transformed_point_cloud.colors.shape if transformed_point_cloud.has_colors else None}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("apply_transform_to_point_cloud_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-source_point_cloud")
    datatypes.visualize(
        transformed_point_cloud, entity_path="/2-transformed_point_cloud"
    )


if __name__ == "__main__":
    apply_transform_to_point_cloud_example()
