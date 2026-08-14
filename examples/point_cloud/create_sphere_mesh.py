"""
Demonstrates creating a UV sphere mesh.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


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

    # ===================== Log ================================================
    logger.success("Created sphere mesh")
    logger.success(f"Results: {sphere_mesh}")
    logger.info(f"Sphere mesh has {len(sphere_mesh)} vertices and {len(sphere_mesh.triangle_indices)} triangles")
    logger.info(f"Sphere mesh has vertex normals: {sphere_mesh.has_vertex_normals()}")
    logger.info(f"Sphere mesh has vertex colors: {sphere_mesh.has_vertex_colors()}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("create_sphere_mesh_example", spawn=True)
    datatypes.visualize(sphere_mesh, entity_path="/SphereMesh/sphere_mesh", label="Sphere Mesh")


if __name__ == "__main__":
    create_sphere_mesh_example()
