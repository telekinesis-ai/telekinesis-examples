"""Demonstrates the Telekinesis Vector3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def vector3d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    vector = [1.0, 2.0, 3.0]
    vector3d = datatypes.Vector3D(vector)

    logger.info(f"Created Vector3D: {vector3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={vector3d.shape}, "
        f"size={vector3d.size}, "
        f"ndim={vector3d.ndim}, "
        f"dtype={vector3d.dtype}"
    )
    logger.info(f"Vector3D data: {vector3d.data}")
    logger.info(f"NumPy array: {vector3d.to_numpy()}")
    logger.info(f"Copied Vector3D: {vector3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("vector3d_example", spawn=True)
    datatypes.visualize(vector3d, entity_path="/Vector3D", label="My Vector3D")

    # ======================= Update ============================================
    new_data = [4.0, 5.0, 6.0]
    vector3d.data = new_data

    logger.info(f"Updated Vector3D: {vector3d}")
    datatypes.visualize(vector3d, entity_path="/Vector3D/updated", label="Updated Vector3D")

    # ======================= NumPy Interop =====================================
    logger.info(f"Sum of Vector3D with numpy array: {vector3d + np.array([1.0, 1.0, 1.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(vector3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Vector3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    vector3d_example()
