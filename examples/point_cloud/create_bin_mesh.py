"""
Demonstrates creating an open-top bin mesh (a floor and four walls).
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def create_bin_mesh_example():
    """
    Creates an open-top bin mesh.

    Generates a bin from its outer length, width and height, with separate
    wall and floor thicknesses. The bin is centered at the origin, with the
    floor at the bottom and the opening facing +Z.
    """
    # ===================== Run Skill ==========================================
    bin_mesh = vitreous.create_bin_mesh(
        transformation_matrix=np.eye(4, dtype=np.float32),
        length=0.4,
        width=0.3,
        height=0.2,
        wall_thickness=0.01,
        floor_thickness=0.01,
        compute_vertex_normals=True,
    )

    # ===================== Log ================================================
    logger.success("Created bin mesh")
    logger.success(f"Results: {bin_mesh}")
    logger.info(
        f"Bin mesh has {len(bin_mesh)} vertices and {len(bin_mesh.triangle_indices)} triangles"
    )
    logger.info(f"Bin mesh has vertex normals: {bin_mesh.has_vertex_normals}")
    logger.info(f"Bin mesh has vertex colors: {bin_mesh.has_vertex_colors}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("create_bin_mesh_example", spawn=True)
    datatypes.visualize(bin_mesh, entity_path="/bin_mesh")


if __name__ == "__main__":
    create_bin_mesh_example()
