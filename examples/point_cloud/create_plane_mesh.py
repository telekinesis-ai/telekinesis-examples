"""
Demonstrates creating a rectangular plane mesh (thin box).

This example:
- Generates a flat rectangular surface with specified dimensions.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous


def create_plane_mesh_example():
    """
    Creates a rectangular plane mesh (thin box).

    Generates a flat rectangular surface with specified dimensions.
    """
    # ===================== Run Skill ==========================================
    plane_mesh = vitreous.create_plane_mesh(
        transformation_matrix=np.eye(4, dtype=np.float32),
        x_dimension=0.01,
        y_dimension=0.01,
        z_dimension=0.00001,
        compute_vertex_normals=True,
    )
    logger.success("Created plane mesh")

    # Access plane_mesh data and properties
    plane_mesh_vertex_positions = plane_mesh.vertex_positions
    plane_mesh_triangle_indices = plane_mesh.triangle_indices
    plane_mesh_vertex_normals = plane_mesh.vertex_normals
    plane_mesh_vertex_colors = plane_mesh.vertex_colors
    logger.info(f"Plane mesh has {len(plane_mesh)} vertices and {len(plane_mesh_triangle_indices)} triangles")
    logger.info(f"Plane mesh has vertex normals: {plane_mesh.has_vertex_normals()}")
    logger.info(f"Plane mesh has vertex colors: {plane_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_plane_mesh_example", spawn=True)
    datatypes.visualize(plane_mesh, entity_path="/PlaneMesh/plane_mesh", label="Plane Mesh")


if __name__ == "__main__":
    create_plane_mesh_example()
