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
    numpy_array = transform2d.to_numpy()

    logger.info(
        f"shape={transform2d.shape}, "
        f"size={transform2d.size}, "
        f"ndim={transform2d.ndim}, "
        f"dtype={transform2d.dtype}"
    )
    logger.info(f"Transform2D data:\n{transform2d.data}")
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Copied Transform2D: {transform2d.copy()}")

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
