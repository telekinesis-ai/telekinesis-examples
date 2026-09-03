"""Demonstrates the Telekinesis Mesh3DBatch datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def mesh3d_batch_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vertex_positions_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    triangle_indices_1 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    vertex_normals_1 = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 1, 1]], dtype=np.float32)
    vertex_colors_1 = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
        dtype=np.uint8,
    )
    mesh_1 = datatypes.Mesh3D(
        vertex_positions=vertex_positions_1,
        triangle_indices=triangle_indices_1,
        vertex_normals=vertex_normals_1,
        vertex_colors=vertex_colors_1,
    )
    mesh_2 = vertex_positions_1 + np.array([5.0, 0.0, 0.0], dtype=np.float32)  # bare ndarray
    mesh_batch = datatypes.Mesh3DBatch([mesh_1, mesh_2])
    logger.info(f"Created Mesh3DBatch: {mesh_batch}")

    # ======================= Inspect ===========================================
    logger.info(f"vertex_positions={mesh_batch.vertex_positions}")
    logger.info(f"triangle_indices={mesh_batch.triangle_indices}")
    logger.info(f"vertex_normals={mesh_batch.vertex_normals}")
    logger.info(f"vertex_colors (packed RGBA uint32)={mesh_batch.vertex_colors}")
    logger.info(f"length={len(mesh_batch)}")

    # ======================= Operations =========================================
    single_mesh = mesh_batch[0]
    logger.info(f"Single Mesh3D at index 0: {single_mesh}")

    sliced_batch = mesh_batch[0:1]
    logger.info(f"Sliced Mesh3DBatch: {sliced_batch}")

    mask = np.array([True, False])
    masked_batch = mesh_batch[mask]
    logger.info(f"Masked Mesh3DBatch: {masked_batch}")

    mesh_batch_copy = mesh_batch.copy()
    logger.info(f"Copied Mesh3DBatch: {mesh_batch_copy}")

    mesh_batch_numpy = mesh_batch.to_numpy(copy=True)
    logger.info(f"NumPy vertex positions per mesh: {[array.shape for array in mesh_batch_numpy]}")

    updated_mesh = datatypes.Mesh3D(
        vertex_positions_1 + np.array([5.0, 0.0, 0.0], dtype=np.float32),
        triangle_indices=triangle_indices_1,
        vertex_colors=np.full((4, 3), [255, 0, 0], dtype=np.uint8),
    )
    rebuilt_batch = datatypes.Mesh3DBatch([mesh_1, updated_mesh])
    logger.info(f"Rebuilt Mesh3DBatch: {rebuilt_batch}")

    # ======================= Visualize =========================================
    rr.init("mesh3d_batch_example", spawn=True)
    datatypes.visualize(
        mesh_batch,
        entity_path="/mesh3d_batch/original",
        label=["Mesh 1", "Mesh 2"],
    )
    datatypes.visualize(
        rebuilt_batch,
        entity_path="/mesh3d_batch/rebuilt",
        label=["Mesh 1", "Updated Mesh 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mesh_batch)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mesh3DBatch: {deserialized}")
    logger.info(f"Round-trip successful: {mesh_batch == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mesh3d_batch_example()
