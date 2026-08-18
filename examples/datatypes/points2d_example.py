"""Demonstrates the Telekinesis Points2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def points2d_example():
    """Demonstrate creation, access, visualization, update, NumPy arithmetic, serialization, and empty-batch construction."""

    # ======================= Create ============================================
    points = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    points2d = datatypes.Points2D(points)

    logger.info(f"Original Points2D: {points2d}")

    # ======================= Inspect ===========================================
    data = points2d.data
    shape = points2d.shape
    size = points2d.size
    dtype = points2d.dtype
    ndim = points2d.ndim
    numpy_points2d = points2d.to_numpy()
    points2d_copy = points2d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_points2d}")
    logger.info(f"Copy: {points2d_copy}")

    # ======================= Visualize =========================================
    rr.init("points2d_example", spawn=True)
    datatypes.visualize(
        points2d, entity_path="/Points2D", label=["Point 1", "Point 2", "Point 3"]
    )

    # ======================= Update ============================================
    new_data = [[70.0, 80.0], [90.0, 100.0], [110.0, 120.0]]
    points2d.data = new_data
    logger.info(f"Updated Points2D: {points2d}")
    datatypes.visualize(
        points2d,
        entity_path="/Points2D/updated",
        label=["Updated Point 1", "Updated Point 2", "Updated Point 3"],
    )

    # ======================= Arithmetic ========================================
    points_sum = points2d + np.array([1.0, 1.0])
    points_diff = points2d - np.array([1.0, 1.0])
    points_prod = points2d * np.array(2.0)
    points_quot = points2d / np.array(2.0)
    logger.info(f"Sum: {points_sum}, Difference: {points_diff}")
    logger.info(f"Product: {points_prod}, Quotient: {points_quot}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(points2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Points2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")

    # ======================= Empty Batch =======================================
    empty = datatypes.Points2D(np.empty((0, 2), dtype=np.float32))
    logger.info(f"Empty Points2D: {empty}, shape={empty.shape}")


if __name__ == "__main__":
    points2d_example()
