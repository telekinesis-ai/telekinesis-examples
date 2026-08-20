"""Demonstrates the Telekinesis Vectors2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def vectors2d_example():
    """Demonstrate creation, access, update, NumPy interop, serialization, and empty batches."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    vectors2d = datatypes.Vectors2D(vectors)

    logger.info(f"Created Vectors2D: {vectors2d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vectors2d.shape}, "
        f"size={vectors2d.size}, "
        f"ndim={vectors2d.ndim}, "
        f"dtype={vectors2d.dtype}"
    )
    logger.info(f"Vectors2D data: {vectors2d.data}")
    logger.info(f"NumPy array: {vectors2d.to_numpy()}")
    logger.info(f"Copied Vectors2D: {vectors2d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vectors2d_example", spawn=True)
    datatypes.visualize(vectors2d, entity_path="/Vectors2D", label=["Vector 1", "Vector 2"])

    # ======================= Update ============================================
    new_data = [[5.0, 6.0], [7.0, 8.0]]
    vectors2d.data = new_data

    logger.info(f"Updated Vectors2D: {vectors2d}")
    datatypes.visualize(
        vectors2d,
        entity_path="/Vectors2D/updated",
        label=["Updated Vector 1", "Updated Vector 2"],
    )

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vectors2D with numpy array: {vectors2d + np.array([1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")

    # ======================= Empty Batch =======================================
    empty = datatypes.Vectors2D(np.empty((0, 2), dtype=np.float32))

    logger.info(f"Empty Vectors2D: {empty}")
    logger.info(f"Empty Vectors2D shape: {empty.shape}")


if __name__ == "__main__":
    vectors2d_example()
