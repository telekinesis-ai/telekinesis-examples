"""Demonstrates the Telekinesis Mesh3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def mesh3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vertex_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    triangle_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    vertex_normals = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 1, 1]], dtype=np.float32)
    vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
        dtype=np.uint8,
    )

    mesh = datatypes.Mesh3D(
        vertex_positions=vertex_positions,
        triangle_indices=triangle_indices,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
    )
    logger.info(f"Created Mesh3D: {mesh}")

    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/gear_box.glb"
    mesh_from_url = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)
    logger.info(f"Mesh3D created from URL: {mesh_from_url}")

    # ======================= Inspect ===========================================
    logger.info(f"vertex_positions={mesh.vertex_positions}")
    logger.info(f"triangle_indices={mesh.triangle_indices}")
    logger.info(f"vertex_normals={mesh.vertex_normals}")
    logger.info(f"vertex_colors (packed RGBA uint32)={mesh.vertex_colors}")
    logger.info(f"has_vertex_normals={mesh.has_vertex_normals}")
    logger.info(f"has_vertex_colors={mesh.has_vertex_colors}")
    logger.info(f"length={len(mesh)}")

    # ======================= Operations =========================================
    mesh_copy = mesh.copy()
    logger.info(f"Copied Mesh3D: {mesh_copy}")

    mesh.save_to_path("results/my_mesh_saved.ply")
    logger.info("Saved Mesh3D to disk as a .ply file.")

    mesh_from_path = datatypes.Mesh3D.from_path("results/my_mesh_saved.ply")
    logger.info(f"Mesh3D loaded from .ply file: {mesh_from_path}")

    # ======================= Visualize =========================================
    rr.init("mesh3d_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/mesh3d", label="My Mesh3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mesh)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mesh3D: {deserialized}")
    logger.info(f"Round-trip successful: {mesh == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mesh3d_example()
