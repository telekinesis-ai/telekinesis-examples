"""Demonstrates the Telekinesis Mat3x3 datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def mat3x3_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    mat3x3 = datatypes.Mat3x3(matrix)
    logger.info(f"Created Mat3x3: {mat3x3}")

    # ======================= Inspect ===========================================
    logger.info(f"data={mat3x3.data}")
    logger.info(f"shape={mat3x3.shape}")
    logger.info(f"ndim={mat3x3.ndim}")
    logger.info(f"dtype={mat3x3.dtype}")
    logger.info(f"size={mat3x3.size}")

    # ======================= Operations =========================================
    mat3x3.data = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    logger.info(f"Updated Mat3x3: {mat3x3}")

    mat3x3_copy = mat3x3.copy()
    logger.info(f"Copied Mat3x3: {mat3x3_copy}")

    mat3x3_numpy = mat3x3.to_numpy(copy=True)
    logger.info(f"NumPy Mat3x3:\n{mat3x3_numpy}")

    numpy_array = np.asarray(mat3x3)
    transposed = np.transpose(mat3x3)
    determinant = np.linalg.det(mat3x3)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Transposed:\n{transposed}")
    logger.info(f"Determinant: {determinant}")

    # ======================= Visualize =========================================
    rr.init("mat3x3_example", spawn=True)
    datatypes.visualize(mat3x3, entity_path="/mat3x3")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mat3x3)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mat3x3: {deserialized}")
    logger.info(f"Round-trip successful: {mat3x3 == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mat3x3_example()
