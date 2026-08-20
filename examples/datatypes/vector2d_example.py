"""Demonstrates the Telekinesis Vector2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def vector2d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    vector = [1.0, 2.0]
    vector2d = datatypes.Vector2D(vector)

    logger.info(f"Created Vector2D: {vector2d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vector2d.shape}, "
        f"size={vector2d.size}, "
        f"ndim={vector2d.ndim}, "
        f"dtype={vector2d.dtype}"
    )
    logger.info(f"Vector2D data: {vector2d.data}")
    logger.info(f"NumPy array: {vector2d.to_numpy()}")
    logger.info(f"Copied Vector2D: {vector2d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vector2d_example", spawn=True)
    datatypes.visualize(vector2d, entity_path="/Vector2D", label="My Vector2D")

    # ======================= Update ============================================
    new_data = [3.0, 4.0]
    vector2d.data = new_data

    logger.info(f"Updated Vector2D: {vector2d}")
    datatypes.visualize(vector2d, entity_path="/Vector2D/updated", label="Updated Vector2D")

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vector2D with numpy array: {vector2d + np.array([1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector2d_example()
