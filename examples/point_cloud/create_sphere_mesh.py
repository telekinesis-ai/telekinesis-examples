"""
Demonstrates creating a UV sphere mesh.

This example:
- Generates a spherical mesh with specified radius and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous


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

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_sphere_mesh_example", spawn=True)
    rr.log("/sphere_mesh", rr.Mesh3D(
        vertex_positions=sphere_mesh.vertex_positions,
        triangle_indices=sphere_mesh.triangle_indices,
        vertex_normals=sphere_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    create_sphere_mesh_example()
