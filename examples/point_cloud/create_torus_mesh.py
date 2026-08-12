"""
Demonstrates creating a torus (donut shape) mesh.

This example:
- Generates a parametric torus with specified major/minor radii and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous


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

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_torus_mesh_example", spawn=True)
    rr.log("/torus_mesh", rr.Mesh3D(
        vertex_positions=torus_mesh.vertex_positions,
        triangle_indices=torus_mesh.triangle_indices,
        vertex_normals=torus_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    create_torus_mesh_example()
