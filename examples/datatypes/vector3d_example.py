"""Demonstrates the Telekinesis Vector3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def vector3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    vector3d = datatypes.Vector3D([1.0, 2.0, 3.0])
    logger.info(f"Created Vector3D: {vector3d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={vector3d.data}")
    logger.info(f"shape={vector3d.shape}")
    logger.info(f"ndim={vector3d.ndim}")
    logger.info(f"dtype={vector3d.dtype}")
    logger.info(f"size={vector3d.size}")

    # ======================= Operations =========================================
    vector3d.data = [4.0, 5.0, 6.0]
    logger.info(f"Updated Vector3D: {vector3d}")

    vector3d_copy = vector3d.copy()
    logger.info(f"Copied Vector3D: {vector3d_copy}")

    vector3d_numpy = vector3d.to_numpy(copy=True)
    logger.info(f"NumPy Vector3D: {vector3d_numpy}")

    numpy_array = np.asarray(vector3d)
    logger.info(f"NumPy array: {numpy_array}")

    sum_with_numpy = vector3d + np.array([1.0, 1.0, 1.0])
    logger.info(f"Sum of Vector3D with NumPy array: {sum_with_numpy}")

    norm = np.linalg.norm(vector3d)
    logger.info(f"Norm (np.linalg.norm): {norm}")

    # ======================= Visualize =========================================
    rr.init("vector3d_example", spawn=True)
    datatypes.visualize(vector3d, entity_path="/vector3d", label="Updated Vector3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector3D: {deserialized}")
    logger.info(f"Round-trip successful: {vector3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector3d_example()
