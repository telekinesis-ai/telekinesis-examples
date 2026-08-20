"""Demonstrates the Telekinesis Point3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def point3d_example():
    """Demonstrate creation, access, visualization, update, NumPy arithmetic, and serialization."""

    # ======================= Create ============================================
    point = [1.0, 2.0, 3.0]
    point3d = datatypes.Point3D(point)

    logger.info(f"Original Point3D: {point3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={point3d.shape}, "
        f"size={point3d.size}, "
        f"ndim={point3d.ndim}, "
        f"dtype={point3d.dtype}"
    )
    logger.info(f"Underlying data: {point3d.data}")
    logger.info(f"NumPy array: {point3d.to_numpy()}")
    logger.info(f"Copy: {point3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("point3d_example", spawn=True)
    datatypes.visualize(point3d, entity_path="/Point3D", label="My Point3D")

    # ======================= Update ============================================
    new_data = [4.0, 5.0, 6.0]
    point3d.data = new_data
    logger.info(f"Updated Point3D: {point3d}")
    datatypes.visualize(point3d, entity_path="/Point3D/updated", label="Updated Point3D")

    # ======================= Arithmetic ========================================
    logger.info(
        f"Sum: {point3d + np.array([1.0, 1.0, 1.0])}, "
        f"Difference: {point3d - np.array([1.0, 1.0, 1.0])}"
    )
    logger.info(f"Product: {point3d * np.array(2.0)}, Quotient: {point3d / np.array(2.0)}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Point3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point3d_example()
