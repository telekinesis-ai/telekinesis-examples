"""Demonstrates the Telekinesis Vector4D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def vector4d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vector4d = datatypes.Vector4D([1.0, 2.0, 3.0, 4.0])
    logger.info(f"Created Vector4D: {vector4d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vector4d.data}")
    logger.info(f"shape={vector4d.shape}")
    logger.info(f"ndim={vector4d.ndim}")
    logger.info(f"dtype={vector4d.dtype}")
    logger.info(f"size={vector4d.size}")

    # ======================= Operations =========================================
    vector4d.data = [5.0, 6.0, 7.0, 8.0]
    logger.info(f"Updated Vector4D: {vector4d}")

    vector4d_copy = vector4d.copy()
    logger.info(f"Copied Vector4D: {vector4d_copy}")

    vector4d_numpy = vector4d.to_numpy(copy=True)
    logger.info(f"NumPy Vector4D: {vector4d_numpy}")

    numpy_array = np.asarray(vector4d)
    logger.info(f"NumPy array: {numpy_array}")

    sum_with_numpy = vector4d + np.array([1.0, 1.0, 1.0, 1.0])
    logger.info(f"Sum of Vector4D with NumPy array: {sum_with_numpy}")

    norm = np.linalg.norm(vector4d)
    logger.info(f"Norm (np.linalg.norm): {norm}")

    # ======================= Visualize =========================================
    # Note: Vector4D has no label support in the visualizer registry.
    rr.init("vector4d_example", spawn=True)
    datatypes.visualize(vector4d, entity_path="/vector4d")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector4d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector4D: {deserialized}")
    logger.info(f"Round-trip successful: {vector4d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector4d_example()
