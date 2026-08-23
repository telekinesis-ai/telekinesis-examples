"""Demonstrates the Telekinesis Vector4D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vector4d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    vector = [1.0, 2.0, 3.0, 4.0]
    vector4d = datatypes.Vector4D(vector)

    logger.info(f"Created Vector4D: {vector4d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vector4d.shape}, "
        f"size={vector4d.size}, "
        f"ndim={vector4d.ndim}, "
        f"dtype={vector4d.dtype}"
    )
    logger.info(f"Vector4D data: {vector4d.data}")
    logger.info(f"NumPy array: {vector4d.to_numpy()}")
    logger.info(f"Copied Vector4D: {vector4d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vector4d_example", spawn=True)
    datatypes.visualize(vector4d, entity_path="/Vector4D")

    # ======================= Update ============================================
    new_data = [5.0, 6.0, 7.0, 8.0]
    vector4d.data = new_data

    logger.info(f"Updated Vector4D: {vector4d}")
    datatypes.visualize(vector4d, entity_path="/Vector4D/updated")

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vector4D with numpy array: {vector4d + np.array([1.0, 1.0, 1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector4d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector4D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector4d_example()
