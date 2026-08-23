"""Demonstrates the Telekinesis Mesh3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def mesh3d_example():
    """Demonstrate construction, access, visualization, loading, saving, and serialization."""

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

    # ======================= Inspect ===========================================
    logger.info(
        f"vertices={len(mesh)}, "
        f"has_vertex_normals={mesh.has_vertex_normals}, "
        f"has_vertex_colors={mesh.has_vertex_colors}"
    )
    logger.info(f"vertex_positions: {mesh.vertex_positions}")
    logger.info(f"triangle_indices: {mesh.triangle_indices}")
    logger.info(f"vertex_normals: {mesh.vertex_normals}")
    logger.info(f"vertex_colors (packed RGBA uint32): {mesh.vertex_colors}")

    # ======================= Visualize =========================================
    rr.init("mesh3d_example", spawn=True)
    rr.log(
        "/Mesh3D/my_mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertex_positions,
            triangle_indices=mesh.triangle_indices,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=mesh.vertex_colors,
        ),
    )

    # ======================= Load From URL =====================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/gear_box.glb"
    mesh_from_url = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    logger.info(f"Mesh3D from URL: {mesh_from_url}")

    # ======================= Save ==============================================
    mesh.save_to_path("results/my_mesh_saved.ply")

    logger.info("Saved Mesh3D to disk as .ply file.")

    # ======================= Load From Path ====================================
    mesh_from_path = datatypes.Mesh3D.from_path("results/my_mesh_saved.ply")

    logger.info(f"Mesh3D from .ply: {mesh_from_path}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mesh)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    arrays_match = (
        np.array_equal(deserialized.vertex_positions, mesh.vertex_positions)
        and np.array_equal(deserialized.triangle_indices, mesh.triangle_indices)
        and np.array_equal(deserialized.vertex_normals, mesh.vertex_normals)
        and np.array_equal(deserialized.vertex_colors, mesh.vertex_colors)
    )

    logger.info(f"Deserialized Mesh3D: {deserialized}")
    logger.info(f"Round-trip successful: {arrays_match}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mesh3d_example()
