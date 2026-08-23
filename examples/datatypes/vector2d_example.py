"""Demonstrates the Telekinesis Vector2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def vector2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vector2d = datatypes.Vector2D([1.0, 2.0])
    logger.info(f"Created Vector2D: {vector2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vector2d.data}")
    logger.info(f"shape={vector2d.shape}")
    logger.info(f"ndim={vector2d.ndim}")
    logger.info(f"dtype={vector2d.dtype}")
    logger.info(f"size={vector2d.size}")

    # ======================= Operations =========================================
    vector2d.data = [3.0, 4.0]
    logger.info(f"Updated Vector2D: {vector2d}")

    vector2d_copy = vector2d.copy()
    logger.info(f"Copied Vector2D: {vector2d_copy}")

    vector2d_numpy = vector2d.to_numpy(copy=True)
    logger.info(f"NumPy Vector2D: {vector2d_numpy}")

    numpy_array = np.asarray(vector2d)
    logger.info(f"NumPy array: {numpy_array}")

    sum_with_numpy = vector2d + np.array([1.0, 1.0])
    logger.info(f"Sum of Vector2D with NumPy array: {sum_with_numpy}")

    norm = np.linalg.norm(vector2d)
    logger.info(f"Norm (np.linalg.norm): {norm}")

    # ======================= Visualize =========================================
    rr.init("vector2d_example", spawn=True)
    datatypes.visualize(vector2d, entity_path="/vector2d", label="Updated Vector2D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector2D: {deserialized}")
    logger.info(f"Round-trip successful: {vector2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector2d_example()
