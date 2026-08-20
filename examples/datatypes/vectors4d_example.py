"""Demonstrates the Telekinesis Vectors4D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def vectors4d_example():
    """Demonstrate creation, access, update, NumPy interop, serialization, and empty batches."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    vectors4d = datatypes.Vectors4D(vectors)

    logger.info(f"Created Vectors4D: {vectors4d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vectors4d.shape}, "
        f"size={vectors4d.size}, "
        f"ndim={vectors4d.ndim}, "
        f"dtype={vectors4d.dtype}"
    )
    logger.info(f"Vectors4D data: {vectors4d.data}")
    logger.info(f"NumPy array: {vectors4d.to_numpy()}")
    logger.info(f"Copied Vectors4D: {vectors4d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vectors4d_example", spawn=True)
    datatypes.visualize(vectors4d, entity_path="/Vectors4D")

    # ======================= Update ============================================
    new_data = [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    vectors4d.data = new_data

    logger.info(f"Updated Vectors4D: {vectors4d}")
    datatypes.visualize(vectors4d, entity_path="/Vectors4D/updated")

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vectors4D with numpy array: {vectors4d + np.array([1.0, 1.0, 1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors4d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors4D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")

    # ======================= Empty Batch =======================================
    empty = datatypes.Vectors4D(np.empty((0, 4), dtype=np.float32))

    logger.info(f"Empty Vectors4D: {empty}")
    logger.info(f"Empty Vectors4D shape: {empty.shape}")


if __name__ == "__main__":
    vectors4d_example()
