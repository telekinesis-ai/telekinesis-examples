"""
Demonstrates converting a triangle mesh to a point cloud via surface sampling.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def convert_mesh_to_point_cloud_example():
    """
    Converts a triangle mesh to a point cloud via surface sampling.

    Samples points on the mesh surface using uniform or Poisson disk sampling.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/gear_box.glb"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    # ===================== Run Skill ==========================================
    point_cloud = vitreous.convert_mesh_to_point_cloud(
        mesh=mesh,
        num_points=10000,
        sampling_method="poisson_disk",
        initial_sampling_factor=5,
        initial_point_cloud=None,
        use_triangle_normal=False,
    )

    # ===================== Log ================================================
    logger.success(f"Converted {mesh} to point cloud")
    logger.success(f"Results: {point_cloud}")
    logger.info(f"Point cloud positions shape: {point_cloud.positions.shape}")
    logger.info(f"Point cloud has normals shape: "
                f"{point_cloud.normals.shape if point_cloud.has_normals else None}")
    logger.info(f"Point cloud has colors shape: "
                f"{point_cloud.colors.shape if point_cloud.has_colors else None}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("convert_mesh_to_point_cloud_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/1-input_mesh", label="Input Mesh")
    datatypes.visualize(point_cloud, entity_path="/2-output_point_cloud", label="Output Point Cloud")


if __name__ == "__main__":
    convert_mesh_to_point_cloud_example()
