"""Demonstrates the Telekinesis Poses2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def poses2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    poses_data = [[1.0, 2.0, 90.0], [3.0, 4.0, 0.0]]
    poses2d = datatypes.Poses2D(poses_data)
    logger.info(f"Created Poses2D: {poses2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={poses2d.data}")
    logger.info(f"shape={poses2d.shape}")
    logger.info(f"ndim={poses2d.ndim}")
    logger.info(f"dtype={poses2d.dtype}")
    logger.info(f"size={poses2d.size}")
    logger.info(f"positions={poses2d.positions}")
    logger.info(f"orientations={poses2d.orientations}")

    # ======================= Operations =========================================
    poses2d.data = [[3.0, 4.0, 45.0], [5.0, 6.0, 180.0]]
    logger.info(f"Updated Poses2D: {poses2d}")

    poses2d_copy = poses2d.copy()
    logger.info(f"Copied Poses2D: {poses2d_copy}")

    poses2d_numpy = poses2d.to_numpy(copy=True)
    logger.info(f"NumPy Poses2D: {poses2d_numpy}")

    first_pose2d = poses2d[0]
    logger.info(f"First Pose2D via indexing: {first_pose2d}")

    poses2d_subset = poses2d[0:1]
    logger.info(f"Poses2D subset via slicing: {poses2d_subset}")

    numpy_array = np.asarray(poses2d)
    translated = numpy_array + np.array([1.0, 1.0, 0.0], dtype=np.float32)
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Translated via NumPy addition:\n{translated}")

    # ======================= Visualize =========================================
    rr.init("poses2d_example", spawn=True)
    datatypes.visualize(poses2d, entity_path="/poses2d", label=["Poses2D 0", "Poses2D 1"])

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(poses2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Poses2D: {deserialized}")
    logger.info(f"Round-trip successful: {poses2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    poses2d_example()
