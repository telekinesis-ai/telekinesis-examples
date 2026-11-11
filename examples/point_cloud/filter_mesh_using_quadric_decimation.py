"""
Demonstrates simplifying a dense mesh to a target triangle count while preserving its shape.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_mesh_using_quadric_decimation_example():
    """
    Simplifies a mesh using quadric error metric decimation.

    Collapses the edges whose removal least changes the surface until the mesh
    has at most the target number of triangles, producing a lighter mesh with
    the same overall shape.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/beer_can.glb"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    # ===================== Run Skill ==========================================
    simplified_mesh = vitreous.filter_mesh_using_quadric_decimation(
        mesh=mesh,
        target_number_of_triangles=5000,
    )

    # ===================== Log ================================================
    logger.success(f"Simplified {mesh} using quadric decimation")
    logger.success(f"Results: {simplified_mesh}")
    logger.info(
        f"Triangles: {len(mesh.triangle_indices)} -> "
        f"{len(simplified_mesh.triangle_indices)}, vertices: "
        f"{len(mesh.vertex_positions)} -> {len(simplified_mesh.vertex_positions)}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_mesh_using_quadric_decimation_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/1-original_mesh")
    datatypes.visualize(simplified_mesh, entity_path="/2-simplified_mesh")


if __name__ == "__main__":
    filter_mesh_using_quadric_decimation_example()
