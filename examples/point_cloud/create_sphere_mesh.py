"""
Demonstrates creating a UV sphere mesh.

This example:
- Generates a spherical mesh with specified radius and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous


def create_sphere_mesh_example():
    """
    Creates a UV sphere mesh.

    Generates a spherical mesh with specified radius and resolution.
    """
    # ===================== Run Skill ==========================================
    sphere_mesh = vitreous.create_sphere_mesh(
        transformation_matrix=np.eye(4, dtype=np.float32),
        radius=0.01,
        resolution=20,
        compute_vertex_normals=True,
    )
    logger.success("Created sphere mesh")

    # Access sphere_mesh data and properties
    sphere_mesh_vertex_positions = sphere_mesh.vertex_positions
    sphere_mesh_triangle_indices = sphere_mesh.triangle_indices
    sphere_mesh_vertex_normals = sphere_mesh.vertex_normals
    sphere_mesh_vertex_colors = sphere_mesh.vertex_colors
    logger.info(f"Sphere mesh has {len(sphere_mesh)} vertices and {len(sphere_mesh_triangle_indices)} triangles")
    logger.info(f"Sphere mesh has vertex normals: {sphere_mesh.has_vertex_normals()}")
    logger.info(f"Sphere mesh has vertex colors: {sphere_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_sphere_mesh_example", spawn=True)
    datatypes.visualize(sphere_mesh, entity_path="/SphereMesh/sphere_mesh", label="Sphere Mesh")

if __name__ == "__main__":
    create_sphere_mesh_example()
