"""Demonstrates the Telekinesis Position2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def position2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    position2d = datatypes.Position2D([10.0, 20.0])
    logger.info(f"Created Position2D: {position2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={position2d.data}")
    logger.info(f"shape={position2d.shape}")
    logger.info(f"ndim={position2d.ndim}")
    logger.info(f"dtype={position2d.dtype}")
    logger.info(f"size={position2d.size}")

    # ======================= Operations =========================================
    position2d.data = [30.0, 40.0]
    logger.info(f"Updated Position2D: {position2d}")

    position2d_copy = position2d.copy()
    logger.info(f"Copied Position2D: {position2d_copy}")

    position2d_numpy = position2d.to_numpy(copy=True)
    logger.info(f"NumPy Position2D: {position2d_numpy}")

    numpy_array = np.asarray(position2d)
    logger.info(f"NumPy array: {numpy_array}")

    sum_with_numpy = position2d + np.array([5.0, 10.0])
    logger.info(f"Sum of Position2D with NumPy array: {sum_with_numpy}")

    distance_from_origin = np.linalg.norm(position2d)
    logger.info(f"Distance from origin (np.linalg.norm): {distance_from_origin}")

    # ======================= Visualize =========================================
    rr.init("position2d_example", spawn=True)
    datatypes.visualize(position2d, entity_path="/position2d", label="Updated Position2D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(position2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Position2D: {deserialized}")
    logger.info(f"Round-trip successful: {position2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    position2d_example()
