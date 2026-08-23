"""Demonstrates the Telekinesis Mat4x4 datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def mat4x4_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    mat4x4 = datatypes.Mat4x4(matrix)
    logger.info(f"Created Mat4x4: {mat4x4}")

    # ======================= Inspect ===========================================
    logger.info(f"data={mat4x4.data}")
    logger.info(f"shape={mat4x4.shape}")
    logger.info(f"ndim={mat4x4.ndim}")
    logger.info(f"dtype={mat4x4.dtype}")
    logger.info(f"size={mat4x4.size}")

    # ======================= Operations =========================================
    mat4x4.data = [
        [16.0, 15.0, 14.0, 13.0],
        [12.0, 11.0, 10.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0, 1.0],
    ]
    logger.info(f"Updated Mat4x4: {mat4x4}")

    mat4x4_copy = mat4x4.copy()
    logger.info(f"Copied Mat4x4: {mat4x4_copy}")

    mat4x4_numpy = mat4x4.to_numpy(copy=True)
    logger.info(f"NumPy Mat4x4:\n{mat4x4_numpy}")

    numpy_array = np.asarray(mat4x4)
    transposed = np.transpose(mat4x4)
    determinant = np.linalg.det(mat4x4)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Transposed:\n{transposed}")
    logger.info(f"Determinant: {determinant}")

    # ======================= Visualize =========================================
    rr.init("mat4x4_example", spawn=True)
    datatypes.visualize(mat4x4, entity_path="/mat4x4")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mat4x4)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mat4x4: {deserialized}")
    logger.info(f"Round-trip successful: {mat4x4 == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mat4x4_example()
