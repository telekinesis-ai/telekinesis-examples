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
    data = point3d.data
    shape = point3d.shape
    size = point3d.size
    dtype = point3d.dtype
    ndim = point3d.ndim
    numpy_point3d = point3d.to_numpy()
    point3d_copy = point3d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_point3d}")
    logger.info(f"Copy: {point3d_copy}")

    # ======================= Visualize =========================================
    rr.init("point3d_example", spawn=True)
    datatypes.visualize(point3d, entity_path="/Point3D", label="My Point3D")

    # ======================= Update ============================================
    new_data = [4.0, 5.0, 6.0]
    point3d.data = new_data
    logger.info(f"Updated Point3D: {point3d}")
    datatypes.visualize(point3d, entity_path="/Point3D/updated", label="Updated Point3D")

    # ======================= Arithmetic ========================================
    point_sum = point3d + np.array([1.0, 1.0, 1.0])
    point_diff = point3d - np.array([1.0, 1.0, 1.0])
    point_prod = point3d * np.array(2.0)
    point_quot = point3d / np.array(2.0)
    logger.info(f"Sum: {point_sum}, Difference: {point_diff}")
    logger.info(f"Product: {point_prod}, Quotient: {point_quot}")

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
