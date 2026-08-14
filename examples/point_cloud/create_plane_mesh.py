"""
Demonstrates creating a rectangular plane mesh (thin box).
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


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

    # ===================== Log ================================================
    logger.success("Created plane mesh")
    logger.success(f"Results: {plane_mesh}")
    logger.info(f"Plane mesh has {len(plane_mesh)} vertices and {len(plane_mesh.triangle_indices)} triangles")
    logger.info(f"Plane mesh has vertex normals: {plane_mesh.has_vertex_normals()}")
    logger.info(f"Plane mesh has vertex colors: {plane_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ===========================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("create_plane_mesh_example", spawn=True)
    datatypes.visualize(plane_mesh, entity_path="/plane_mesh", label="Plane Mesh")


if __name__ == "__main__":
    create_plane_mesh_example()
