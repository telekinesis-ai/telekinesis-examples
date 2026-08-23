"""Demonstrates the Telekinesis Poses2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def poses2d_example():
    """Demonstrate creation, access, visualization, update, NumPy translation, and serialization."""

    # ======================= Create ============================================
    poses = [[1.0, 2.0, 0.5], [3.0, 4.0, 1.2]]
    poses2d = datatypes.Poses2D(poses)

    logger.info(f"Original Poses2D: {poses2d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={poses2d.shape}, "
        f"size={poses2d.size}, "
        f"ndim={poses2d.ndim}, "
        f"dtype={poses2d.dtype}"
    )
    logger.info(f"Underlying data: {poses2d.data}")
    logger.info(f"NumPy array: {poses2d.to_numpy()}")
    logger.info(f"Copy: {poses2d.copy()}")

    # ======================= Visualize =========================================
    rr.init("poses2d_example", spawn=True)
    datatypes.visualize(poses2d, entity_path="/Poses2D", label=["My Poses2D 0", "My Poses2D 1"])

    # ======================= Update ============================================
    new_data = [[3.0, 4.0, 1.0], [5.0, 6.0, 2.0]]
    poses2d.data = new_data
    logger.info(f"Updated Poses2D: {poses2d}")
    datatypes.visualize(
        poses2d,
        entity_path="/Poses2D/updated",
        label=["Updated Poses2D 0", "Updated Poses2D 1"],
    )

    # ======================= Translate =========================================
    logger.info(f"Translated Poses2D: {poses2d + np.array([1.0, 1.0, 0.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(poses2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Poses2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    poses2d_example()
