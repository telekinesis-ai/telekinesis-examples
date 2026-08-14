"""
Demonstrates reconstructing a watertight mesh from an oriented point cloud using Poisson surface reconstruction.
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

    # ===================== Run Skill ==========================================
    reconstructed_mesh = vitreous.reconstruct_mesh_using_poisson(
        octree_depth=7,
        octree_width=0,
        scale_factor=1.1,
        point_cloud=point_cloud,
    )

    # ===================== Log ================================================
    logger.success(f"Reconstructed mesh from {point_cloud} using Poisson")
    logger.success(f"Results: {reconstructed_mesh}")
    logger.info(
        f"Reconstructed mesh has {len(reconstructed_mesh.vertex_positions)} vertices and {len(reconstructed_mesh.triangle_indices)} triangles"
    )
    logger.info(
        f"Reconstructed mesh has vertex normals: {reconstructed_mesh.has_vertex_normals}"
    )
    logger.info(
        f"Reconstructed mesh has vertex colors: {reconstructed_mesh.has_vertex_colors}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("reconstruct_mesh_using_poisson_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(reconstructed_mesh, entity_path="/2-poisson_mesh")


if __name__ == "__main__":
    reconstruct_mesh_using_poisson_example()
