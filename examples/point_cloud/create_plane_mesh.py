"""
Demonstrates creating a rectangular plane mesh (thin box).

This example:
- Generates a flat rectangular surface with specified dimensions.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous


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

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_plane_mesh_example", spawn=True)
    rr.log("/plane_mesh", rr.Mesh3D(
        vertex_positions=plane_mesh.vertex_positions,
        triangle_indices=plane_mesh.triangle_indices,
        vertex_normals=plane_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    create_plane_mesh_example()
