"""
Demonstrates creating a torus (donut shape) mesh.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


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

    # ===================== Log ================================================
    logger.success("Created torus mesh")
    logger.success(f"Results: {torus_mesh}")
    logger.info(
        f"Torus mesh has {len(torus_mesh)} vertices and {len(torus_mesh.triangle_indices)} triangles"
    )
    logger.info(f"Torus mesh has vertex normals: {torus_mesh.has_vertex_normals}")
    logger.info(f"Torus mesh has vertex colors: {torus_mesh.has_vertex_colors}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("create_torus_mesh_example", spawn=True)
    datatypes.visualize(torus_mesh, entity_path="/torus_mesh")


if __name__ == "__main__":
    create_torus_mesh_example()
