"""
Demonstrates projecting a point cloud orthogonally onto a plane.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def project_point_cloud_to_plane_example():
    """
    Projects all points orthogonally onto a plane.

    Moves each point to its closest point on the specified plane. Flattens
    the cloud onto a 2D surface in 3D space.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/engine_parts_0.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    projected_point_cloud = vitreous.project_point_cloud_to_plane(
        add_white_noise=False,
        white_noise_standard_deviation=1e-6,
        point_cloud=point_cloud,
        plane_coefficients=[0.0, 0.0, 1.0, 0.0],
    )

    # ===================== Log ================================================
    logger.success(f"Projected {point_cloud} to plane")
    logger.success(f"Results: {projected_point_cloud}")
    logger.info(f"Projected point cloud positions shape: {projected_point_cloud.positions.shape}")
    logger.info(f"Projected point cloud has normals shape: "
                f"{projected_point_cloud.normals.shape if projected_point_cloud.has_normals else None}")
    logger.info(f"Projected point cloud has colors shape: "
                f"{projected_point_cloud.colors.shape if projected_point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("project_point_cloud_to_plane_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(projected_point_cloud, entity_path="/2-filtered_point_cloud")


if __name__ == "__main__":
    project_point_cloud_to_plane_example()
