"""
Demonstrates removing the base faces from a cylindrical mesh, leaving only the curved side surface.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_cylinder_base_removal_example():
    """
    Removes the base faces from a cylindrical mesh.

    Identifies and removes triangles that form the flat base(s) of a cylinder,
    leaving only the curved side surface.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/beer_can.glb"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    # ===================== Run Skill ==========================================
    filtered_mesh = vitreous.filter_point_cloud_using_cylinder_base_removal(
        mesh=mesh,
        compute_vertex_normals=True,
        distance_threshold=0.005,
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {mesh} using cylinder base removal")
    logger.success(f"Results: {filtered_mesh}")
    logger.info(
        f"Filtered mesh has {len(filtered_mesh.vertex_positions)} vertices "
        f"and {len(filtered_mesh.triangle_indices)} triangles"
    )
    logger.info(
        f"Filtered mesh has vertex normals: {filtered_mesh.has_vertex_normals}, "
        f"vertex colors: {filtered_mesh.has_vertex_colors}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_cylinder_base_removal_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/1-original_mesh")
    datatypes.visualize(filtered_mesh, entity_path="/2-filtered_mesh")


if __name__ == "__main__":
    filter_point_cloud_using_cylinder_base_removal_example()
