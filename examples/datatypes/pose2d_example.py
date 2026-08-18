"""Demonstrates the Telekinesis Pose2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def pose2d_example():
    """Demonstrate creation, access, visualization, update, NumPy translation, and serialization."""

    # ======================= Create ============================================
    pose = [1.0, 2.0, 0.5]
    pose2d = datatypes.Pose2D(pose)

    logger.info(f"Original Pose2D: {pose2d}")

    # ======================= Inspect ===========================================
    data = pose2d.data
    shape = pose2d.shape
    size = pose2d.size
    dtype = pose2d.dtype
    ndim = pose2d.ndim
    numpy_pose2d = pose2d.to_numpy()
    pose2d_copy = pose2d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_pose2d}")
    logger.info(f"Copy: {pose2d_copy}")

    # ======================= Visualize =========================================
    rr.init("pose2d_example", spawn=True)
    datatypes.visualize(pose2d, entity_path="/Pose2D", label="My Pose2D")

    # ======================= Update ============================================
    new_data = [3.0, 4.0, 1.0]
    pose2d.data = new_data
    logger.info(f"Updated Pose2D: {pose2d}")
    datatypes.visualize(pose2d, entity_path="/Pose2D/updated", label="Updated Pose2D")

    # ======================= Translate =========================================
    translated = pose2d + np.array([1.0, 1.0, 0.0])
    logger.info(f"Translated Pose2D: {translated}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(pose2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Pose2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == new_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    pose2d_example()
