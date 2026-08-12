"""
Demonstrates projecting a point cloud orthogonally onto a plane.

This example:
- Downloads an example point cloud.
- Projects all points orthogonally onto a plane, flattening the cloud onto a 2D surface in 3D space.
- Visualizes the result using Rerun.
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
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    projected_point_cloud = vitreous.project_point_cloud_to_plane(
        add_white_noise=False,
        white_noise_standard_deviation=1e-6,
        point_cloud=point_cloud,
        plane_coefficients=[0.0, 0.0, 1.0, 0.0],
    )
    logger.success(f"Projected {len(projected_point_cloud)} points to plane")

    # ===================== Visualization  (Optional) ======================
    rr.init("project_point_cloud_to_plane_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(projected_point_cloud, entity_path="/projected_point_cloud")


if __name__ == "__main__":
    project_point_cloud_to_plane_example()
