"""Demonstrates the Telekinesis Position3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def position3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    position3d = datatypes.Position3D([1.0, 2.0, 3.0])
    logger.info(f"Created Position3D: {position3d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={position3d.data}")
    logger.info(f"shape={position3d.shape}")
    logger.info(f"ndim={position3d.ndim}")
    logger.info(f"dtype={position3d.dtype}")
    logger.info(f"size={position3d.size}")

    # ======================= Operations =========================================
    position3d.data = [4.0, 5.0, 6.0]
    logger.info(f"Updated Position3D: {position3d}")

    position3d_copy = position3d.copy()
    logger.info(f"Copied Position3D: {position3d_copy}")

    position3d_numpy = position3d.to_numpy(copy=True)
    logger.info(f"NumPy Position3D: {position3d_numpy}")

    numpy_array = np.asarray(position3d)
    logger.info(f"NumPy array: {numpy_array}")

    difference_with_numpy = position3d - np.array([1.0, 1.0, 1.0])
    logger.info(f"Difference of Position3D with NumPy array: {difference_with_numpy}")

    distance_from_origin = np.linalg.norm(position3d)
    logger.info(f"Distance from origin (np.linalg.norm): {distance_from_origin}")

    # ======================= Visualize =========================================
    rr.init("position3d_example", spawn=True)
    datatypes.visualize(position3d, entity_path="/position3d", label="Updated Position3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(position3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Position3D: {deserialized}")
    logger.info(f"Round-trip successful: {position3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    position3d_example()
