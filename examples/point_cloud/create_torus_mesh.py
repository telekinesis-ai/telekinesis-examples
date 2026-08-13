"""
Demonstrates creating a torus (donut shape) mesh.

This example:
- Generates a parametric torus with specified major/minor radii and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous


def create_torus_mesh_example():
    """
    Creates a torus (donut shape) mesh.

    Generates a parametric torus with specified major/minor radii and resolution.
    """
    # ===================== Run Skill ==========================================
    torus_mesh = vitreous.create_torus_mesh(
        transformation_matrix=np.eye(4, dtype=np.float32),
        torus_radius=0.01,
        tube_radius=0.005,
        radial_resolution=20,
        tubular_resolution=10,
        compute_vertex_normals=True,
    )
    logger.success("Created torus mesh")

    # Access torus_mesh data and properties
    torus_mesh_vertex_positions = torus_mesh.vertex_positions
    torus_mesh_triangle_indices = torus_mesh.triangle_indices
    torus_mesh_vertex_normals = torus_mesh.vertex_normals
    torus_mesh_vertex_colors = torus_mesh.vertex_colors
    logger.info(f"Torus mesh has {len(torus_mesh)} vertices and {len(torus_mesh_triangle_indices)} triangles")
    logger.info(f"Torus mesh has vertex normals: {torus_mesh.has_vertex_normals()}")
    logger.info(f"Torus mesh has vertex colors: {torus_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_torus_mesh_example", spawn=True)
    datatypes.visualize(torus_mesh, entity_path="/TorusMesh/torus_mesh", label="Torus Mesh")


if __name__ == "__main__":
    create_torus_mesh_example()
