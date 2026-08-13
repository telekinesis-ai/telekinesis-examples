"""
Demonstrates projecting a point cloud onto a plane defined by a point and normal.

This example:
- Downloads an example point cloud.
- Projects points onto a plane defined by a point and normal (alternative parameterization).
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def project_point_cloud_to_plane_defined_by_point_normal_example():
    """
    Projects points onto a plane defined by a point and normal (alternative parameterization).

    Same as plane projection but using point+normal instead of coefficients.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    projected_point_cloud = vitreous.project_point_cloud_to_plane_defined_by_point_normal(
        add_white_noise=False,
        white_noise_standard_deviation=1e-6,
        point_cloud=point_cloud,
        point=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
    )
    logger.success(f"Projected {len(projected_point_cloud)} points to plane (point+normal)")

    # Access projected_point_cloud data and properties
    projected_point_cloud_positions = projected_point_cloud.positions
    projected_point_cloud_normals = projected_point_cloud.normals
    projected_point_cloud_colors = projected_point_cloud.colors
    logger.info(f"Projected point cloud positions shape: {projected_point_cloud_positions.shape}")
    logger.info(f"Projected point cloud has normals: {projected_point_cloud_normals is not None}")
    logger.info(f"Projected point cloud has colors: {projected_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("project_point_cloud_to_plane_defined_by_point_normal_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(projected_point_cloud, entity_path="/projected_point_cloud")


if __name__ == "__main__":
    project_point_cloud_to_plane_defined_by_point_normal_example()
