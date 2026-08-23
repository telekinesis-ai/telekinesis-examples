"""Demonstrates the Telekinesis Vectors3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vectors3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    vectors3d = datatypes.Vectors3D(vectors)
    logger.info(f"Created Vectors3D: {vectors3d}")

    empty_vectors3d = datatypes.Vectors3D(np.empty((0, 3), dtype=np.float32))
    logger.info(f"Created empty Vectors3D batch: {empty_vectors3d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vectors3d.data}")
    logger.info(f"shape={vectors3d.shape}")
    logger.info(f"ndim={vectors3d.ndim}")
    logger.info(f"dtype={vectors3d.dtype}")
    logger.info(f"size={vectors3d.size}")

    # ======================= Operations =========================================
    vectors3d.data = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    logger.info(f"Updated Vectors3D: {vectors3d}")

    vectors3d_copy = vectors3d.copy()
    logger.info(f"Copied Vectors3D: {vectors3d_copy}")

    vectors3d_numpy = vectors3d.to_numpy(copy=True)
    logger.info(f"NumPy Vectors3D:\n{vectors3d_numpy}")

    numpy_array = np.asarray(vectors3d)
    column_sums = np.sum(vectors3d, axis=0)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Column sums: {column_sums}")

    # ======================= Visualize =========================================
    rr.init("vectors3d_example", spawn=True)
    datatypes.visualize(vectors3d, entity_path="/vectors3d", label=["Vector 1", "Vector 2"])

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors3D: {deserialized}")
    logger.info(f"Round-trip successful: {vectors3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vectors3d_example()
