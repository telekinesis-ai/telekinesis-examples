"""Demonstrates the Telekinesis Transform2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def transform2d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    theta = np.pi / 4
    matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 1.0],
            [np.sin(theta), np.cos(theta), 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform2d = datatypes.Transform2D(matrix)
    logger.info(f"Original Transform2D: {transform2d}")

    # ======================= Inspect ===========================================
    data = transform2d.data
    shape = transform2d.shape
    size = transform2d.size
    dtype = transform2d.dtype
    ndim = transform2d.ndim
    numpy_array = transform2d.to_numpy()
    copy = transform2d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Transform2D data:\n{data}")
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Copied Transform2D: {copy}")

    # ======================= Visualize =========================================
    rr.init("transform2d_example", spawn=True)
    datatypes.visualize(
        transform2d,
        entity_path="/Transform2D/main",
        label="My Transform2D",
    )

    # ======================= Update ============================================
    updated_theta = np.pi / 2
    updated_matrix = np.array(
        [
            [np.cos(updated_theta), -np.sin(updated_theta), 1.5],
            [np.sin(updated_theta), np.cos(updated_theta), 2.5],
            [0.0, 0.0, 1.0],
        ]
    )
    transform2d.data = updated_matrix
    logger.info(f"Updated Transform2D: {transform2d}")
    datatypes.visualize(
        transform2d,
        entity_path="/Transform2D/updated",
        label="Updated Transform2D",
    )

    # ======================= Arithmetic ========================================
    total = np.array([1, 1, 0]) + numpy_array
    logger.info(f"Sum of Transform2D with numpy array: {total}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(transform2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Transform2D: {deserialized}")
    logger.info(f"Round-trip successful: {transform2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    transform2d_example()
