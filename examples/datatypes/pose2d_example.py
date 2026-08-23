"""Demonstrates the Telekinesis Pose2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def pose2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    pose_data = [1.0, 2.0, 90.0]
    pose2d = datatypes.Pose2D(pose_data)
    logger.info(f"Created Pose2D: {pose2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data={pose2d.data}")
    logger.info(f"shape={pose2d.shape}")
    logger.info(f"ndim={pose2d.ndim}")
    logger.info(f"dtype={pose2d.dtype}")
    logger.info(f"size={pose2d.size}")
    logger.info(f"position={pose2d.position}")
    logger.info(f"orientation={pose2d.orientation}")

    # ======================= Operations =========================================
    pose2d.data = [3.0, 4.0, 45.0]
    logger.info(f"Updated Pose2D: {pose2d}")

    pose2d_copy = pose2d.copy()
    logger.info(f"Copied Pose2D: {pose2d_copy}")

    pose2d_numpy = pose2d.to_numpy(copy=True)
    logger.info(f"NumPy Pose2D: {pose2d_numpy}")

    transform2d = pose2d.to_transform2d()
    logger.info(f"Pose2D as Transform2D: {transform2d}")

    numpy_array = np.asarray(pose2d)
    translated = numpy_array + np.array([1.0, 1.0, 0.0], dtype=np.float32)
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Translated via NumPy addition: {translated}")

    # ======================= Visualize =========================================
    rr.init("pose2d_example", spawn=True)
    datatypes.visualize(pose2d, entity_path="/pose2d", label="Pose2D")
    datatypes.visualize(
        transform2d, entity_path="/pose2d/transform2d", label="Pose2D As Transform2D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(pose2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Pose2D: {deserialized}")
    logger.info(f"Round-trip successful: {pose2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    pose2d_example()
