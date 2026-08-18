"""Demonstrates the Telekinesis Points3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def points3d_example():
    """Demonstrate creation, access, visualization, update, NumPy arithmetic, serialization, and empty-batch construction."""

    # ======================= Create ============================================
    points = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    points3d = datatypes.Points3D(points)

    logger.info(f"Original Points3D: {points3d}")

    # ======================= Inspect ===========================================
    data = points3d.data
    shape = points3d.shape
    size = points3d.size
    dtype = points3d.dtype
    ndim = points3d.ndim
    numpy_points3d = points3d.to_numpy()
    points3d_copy = points3d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_points3d}")
    logger.info(f"Copy: {points3d_copy}")

    # ======================= Visualize =========================================
    rr.init("points3d_example", spawn=True)
    datatypes.visualize(points3d, entity_path="/Points3D", label=["Point 1", "Point 2"])

    # ======================= Update ============================================
    new_data = [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]
    points3d.data = new_data
    logger.info(f"Updated Points3D: {points3d}")
    datatypes.visualize(
        points3d,
        entity_path="/Points3D/updated",
        label=["Updated Point 1", "Updated Point 2"],
    )

    # ======================= Arithmetic ========================================
    points_sum = points3d + np.array([1.0, 1.0, 1.0])
    points_diff = points3d - np.array([1.0, 1.0, 1.0])
    points_prod = points3d * np.array(2.0)
    points_quot = points3d / np.array(2.0)
    logger.info(f"Sum: {points_sum}, Difference: {points_diff}")
    logger.info(f"Product: {points_prod}, Quotient: {points_quot}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(points3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Points3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")

    # ======================= Empty Batch =======================================
    empty = datatypes.Points3D(np.empty((0, 3), dtype=np.float32))
    logger.info(f"Empty Points3D: {empty}, shape={empty.shape}")


if __name__ == "__main__":
    points3d_example()
