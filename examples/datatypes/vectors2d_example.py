"""Demonstrates the Telekinesis Vectors2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vectors2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    vectors2d = datatypes.Vectors2D(vectors)
    logger.info(f"Created Vectors2D: {vectors2d}")

    empty_vectors2d = datatypes.Vectors2D(np.empty((0, 2), dtype=np.float32))
    logger.info(f"Created empty Vectors2D batch: {empty_vectors2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vectors2d.data}")
    logger.info(f"shape={vectors2d.shape}")
    logger.info(f"ndim={vectors2d.ndim}")
    logger.info(f"dtype={vectors2d.dtype}")
    logger.info(f"size={vectors2d.size}")

    # ======================= Operations =========================================
    vectors2d.data = [[5.0, 6.0], [7.0, 8.0]]
    logger.info(f"Updated Vectors2D: {vectors2d}")

    vectors2d_copy = vectors2d.copy()
    logger.info(f"Copied Vectors2D: {vectors2d_copy}")

    vectors2d_numpy = vectors2d.to_numpy(copy=True)
    logger.info(f"NumPy Vectors2D:\n{vectors2d_numpy}")

    numpy_array = np.asarray(vectors2d)
    column_sums = np.sum(vectors2d, axis=0)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Column sums: {column_sums}")

    # ======================= Visualize =========================================
    rr.init("vectors2d_example", spawn=True)
    datatypes.visualize(vectors2d, entity_path="/vectors2d", label=["Vector 1", "Vector 2"])

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors2D: {deserialized}")
    logger.info(f"Round-trip successful: {vectors2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vectors2d_example()
