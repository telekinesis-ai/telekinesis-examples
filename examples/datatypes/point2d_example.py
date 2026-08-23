"""Demonstrates the Telekinesis Point2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def point2d_example():
    """Demonstrate creation, access, visualization, update, NumPy arithmetic, and serialization."""

    # ======================= Create ============================================
    point = [1.0, 2.0]
    point2d = datatypes.Point2D(point)

    logger.info(f"Original Point2D: {point2d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={point2d.shape}, "
        f"size={point2d.size}, "
        f"ndim={point2d.ndim}, "
        f"dtype={point2d.dtype}"
    )
    logger.info(f"Underlying data: {point2d.data}")
    logger.info(f"NumPy array: {point2d.to_numpy()}")
    logger.info(f"Copy: {point2d.copy()}")

    # ======================= Visualize =========================================
    rr.init("point2d_example", spawn=True)
    datatypes.visualize(point2d, entity_path="/Point2D", label="My Point2D")

    # ======================= Update ============================================
    new_data = [3.0, 4.0]
    point2d.data = new_data
    logger.info(f"Updated Point2D: {point2d}")
    datatypes.visualize(point2d, entity_path="/Point2D/updated", label="Updated Point2D")

    # ======================= Arithmetic ========================================
    logger.info(
        f"Sum: {point2d + np.array([1.0, 1.0])}, "
        f"Difference: {point2d - np.array([1.0, 1.0])}"
    )
    logger.info(f"Product: {point2d * np.array(2.0)}, Quotient: {point2d / np.array(2.0)}")

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
