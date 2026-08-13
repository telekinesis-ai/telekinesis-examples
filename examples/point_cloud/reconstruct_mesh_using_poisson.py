"""
Demonstrates reconstructing a watertight mesh from an oriented point cloud using Poisson surface reconstruction.

This example:
- Downloads an example point cloud.
- Solves a Poisson equation to fit a smooth, closed surface through points with normals.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def reconstruct_mesh_using_poisson_example():
    """
    Reconstructs a watertight mesh from an oriented point cloud using Poisson surface reconstruction.

    Solves a Poisson equation to fit a smooth surface through points with normals.
    Produces closed, manifold meshes. Requires point cloud normals.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/industrial_part_7_normals.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    reconstructed_mesh = vitreous.reconstruct_mesh_using_poisson(
        octree_depth=7, octree_width=0, scale_factor=1.1,
        point_cloud=point_cloud,
    )
    logger.success(f"Reconstructed mesh from {len(point_cloud)} points using Poisson")

    # Access reconstructed_mesh data and properties
    reconstructed_mesh_vertex_positions = reconstructed_mesh.vertex_positions
    reconstructed_mesh_triangle_indices = reconstructed_mesh.triangle_indices
    reconstructed_mesh_vertex_normals = reconstructed_mesh.vertex_normals
    reconstructed_mesh_vertex_colors = reconstructed_mesh.vertex_colors
    logger.info(f"Reconstructed mesh has {len(reconstructed_mesh)} vertices and {len(reconstructed_mesh_triangle_indices)} triangles")
    logger.info(f"Reconstructed mesh has vertex normals: {reconstructed_mesh.has_vertex_normals()}")
    logger.info(f"Reconstructed mesh has vertex colors: {reconstructed_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("reconstruct_mesh_using_poisson_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    rr.log("/poisson_mesh", rr.Mesh3D(
        vertex_positions=reconstructed_mesh.vertex_positions,
        triangle_indices=reconstructed_mesh.triangle_indices,
        vertex_normals=reconstructed_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    reconstruct_mesh_using_poisson_example()
