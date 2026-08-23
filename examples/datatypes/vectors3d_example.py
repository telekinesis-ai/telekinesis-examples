"""Demonstrates the Telekinesis Vectors3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vectors3d_example():
    """Demonstrate creation, access, update, NumPy interop, serialization, and empty batches."""

    # ======================= Create ============================================
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    vectors3d = datatypes.Vectors3D(vectors)

    logger.info(f"Created Vectors3D: {vectors3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vectors3d.shape}, "
        f"size={vectors3d.size}, "
        f"ndim={vectors3d.ndim}, "
        f"dtype={vectors3d.dtype}"
    )
    logger.info(f"Vectors3D data: {vectors3d.data}")
    logger.info(f"NumPy array: {vectors3d.to_numpy()}")
    logger.info(f"Copied Vectors3D: {vectors3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vectors3d_example", spawn=True)
    datatypes.visualize(vectors3d, entity_path="/Vectors3D", label=["Vector 1", "Vector 2"])

    # ======================= Update ============================================
    new_data = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    vectors3d.data = new_data

    logger.info(f"Updated Vectors3D: {vectors3d}")
    datatypes.visualize(
        vectors3d,
        entity_path="/Vectors3D/updated",
        label=["Updated Vector 1", "Updated Vector 2"],
    )

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vectors3D with numpy array: {vectors3d + np.array([1.0, 1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vectors3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vectors3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")

    # ======================= Empty Batch =======================================
    empty = datatypes.Vectors3D(np.empty((0, 3), dtype=np.float32))

    logger.info(f"Empty Vectors3D: {empty}")
    logger.info(f"Empty Vectors3D shape: {empty.shape}")


if __name__ == "__main__":
    vectors3d_example()
