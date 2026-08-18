"""Demonstrates the Telekinesis Point2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def point2d_example():
    """Demonstrate creation, access, visualization, update, NumPy arithmetic, and serialization."""

    # ======================= Create ============================================
    point = [1.0, 2.0]
    point2d = datatypes.Point2D(point)

    logger.info(f"Original Point2D: {point2d}")

    # ======================= Inspect ===========================================
    data = point2d.data
    shape = point2d.shape
    size = point2d.size
    dtype = point2d.dtype
    ndim = point2d.ndim
    numpy_point2d = point2d.to_numpy()
    point2d_copy = point2d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_point2d}")
    logger.info(f"Copy: {point2d_copy}")

    # ======================= Visualize =========================================
    rr.init("point2d_example", spawn=True)
    datatypes.visualize(point2d, entity_path="/Point2D", label="My Point2D")

    # ======================= Update ============================================
    new_data = [3.0, 4.0]
    point2d.data = new_data
    logger.info(f"Updated Point2D: {point2d}")
    datatypes.visualize(point2d, entity_path="/Point2D/updated", label="Updated Point2D")

    # ======================= Arithmetic ========================================
    point_sum = point2d + np.array([1.0, 1.0])
    point_diff = point2d - np.array([1.0, 1.0])
    point_prod = point2d * np.array(2.0)
    point_quot = point2d / np.array(2.0)
    logger.info(f"Sum: {point_sum}, Difference: {point_diff}")
    logger.info(f"Product: {point_prod}, Quotient: {point_quot}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Point2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point2d_example()
