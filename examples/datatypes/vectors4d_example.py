"""Demonstrates the Telekinesis Vectors4D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vectors4d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    vectors4d = datatypes.Vectors4D(vectors)
    logger.info(f"Created Vectors4D: {vectors4d}")

    empty_vectors4d = datatypes.Vectors4D(np.empty((0, 4), dtype=np.float32))
    logger.info(f"Created empty Vectors4D batch: {empty_vectors4d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vectors4d.data}")
    logger.info(f"shape={vectors4d.shape}")
    logger.info(f"ndim={vectors4d.ndim}")
    logger.info(f"dtype={vectors4d.dtype}")
    logger.info(f"size={vectors4d.size}")

    # ======================= Operations =========================================
    vectors4d.data = [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    logger.info(f"Updated Vectors4D: {vectors4d}")

    vectors4d_copy = vectors4d.copy()
    logger.info(f"Copied Vectors4D: {vectors4d_copy}")

    vectors4d_numpy = vectors4d.to_numpy(copy=True)
    logger.info(f"NumPy Vectors4D:\n{vectors4d_numpy}")

    numpy_array = np.asarray(vectors4d)
    column_sums = np.sum(vectors4d, axis=0)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Column sums: {column_sums}")

    # ======================= Visualize =========================================
    rr.init("vectors4d_example", spawn=True)
    datatypes.visualize(vectors4d, entity_path="/vectors4d")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors4d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors4D: {deserialized}")
    logger.info(f"Round-trip successful: {vectors4d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vectors4d_example()
