"""
Demonstrates creating a parametric cylinder mesh.

This example:
- Generates a cylinder with specified radius, height, and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def create_cylinder_mesh_example():
    """
    Creates a parametric cylinder mesh.

    Generates a cylinder with specified radius, height, and resolution.
    """
    # ===================== Run Skill ==========================================
    cylinder_mesh = vitreous.create_cylinder_mesh(
        radius=0.01,
        height=0.02,
        radial_resolution=20,
        height_resolution=4,
        retain_base=False,
        vertex_tolerance=1e-6,
        transformation_matrix=np.eye(4, dtype=np.float32),
        compute_vertex_normals=True,
    )
    logger.success("Created cylinder mesh")

    # Access cylinder_mesh data and properties
    cylinder_mesh_vertex_positions = cylinder_mesh.vertex_positions
    cylinder_mesh_triangle_indices = cylinder_mesh.triangle_indices
    cylinder_mesh_vertex_normals = cylinder_mesh.vertex_normals
    cylinder_mesh_vertex_colors = cylinder_mesh.vertex_colors
    logger.info(f"Cylinder mesh has {len(cylinder_mesh)} vertices and {len(cylinder_mesh_triangle_indices)} triangles")
    logger.info(f"Cylinder mesh has vertex normals: {cylinder_mesh.has_vertex_normals()}")
    logger.info(f"Cylinder mesh has vertex colors: {cylinder_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_cylinder_mesh_example", spawn=True)
    datatypes.visualize(cylinder_mesh, entity_path="/CylinderMesh/cylinder_mesh", label="Cylinder Mesh")


if __name__ == "__main__":
    create_cylinder_mesh_example()
