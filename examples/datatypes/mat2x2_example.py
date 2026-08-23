"""Demonstrates the Telekinesis Mat2x2 datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def mat2x2_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    mat2x2 = datatypes.Mat2x2(matrix)
    logger.info(f"Created Mat2x2: {mat2x2}")

    # ======================= Inspect ===========================================
    logger.info(f"data={mat2x2.data}")
    logger.info(f"shape={mat2x2.shape}")
    logger.info(f"ndim={mat2x2.ndim}")
    logger.info(f"dtype={mat2x2.dtype}")
    logger.info(f"size={mat2x2.size}")

    # ======================= Operations =========================================
    mat2x2.data = [[5.0, 6.0], [7.0, 8.0]]
    logger.info(f"Updated Mat2x2: {mat2x2}")

    mat2x2_copy = mat2x2.copy()
    logger.info(f"Copied Mat2x2: {mat2x2_copy}")

    mat2x2_numpy = mat2x2.to_numpy(copy=True)
    logger.info(f"NumPy Mat2x2:\n{mat2x2_numpy}")

    numpy_array = np.asarray(mat2x2)
    transposed = np.transpose(mat2x2)
    determinant = np.linalg.det(mat2x2)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Transposed:\n{transposed}")
    logger.info(f"Determinant: {determinant}")

    # ======================= Visualize =========================================
    rr.init("mat2x2_example", spawn=True)
    datatypes.visualize(mat2x2, entity_path="/mat2x2")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mat2x2)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mat2x2: {deserialized}")
    logger.info(f"Round-trip successful: {mat2x2 == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mat2x2_example()
