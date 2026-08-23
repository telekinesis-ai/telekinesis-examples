"""Demonstrates the Telekinesis Transform2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def transform2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

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
    logger.info(f"Created Transform2D: {transform2d}")

    # ======================= Inspect ===========================================
    logger.info(f"data=\n{transform2d.data}")
    logger.info(f"shape={transform2d.shape}")
    logger.info(f"ndim={transform2d.ndim}")
    logger.info(f"dtype={transform2d.dtype}")
    logger.info(f"size={transform2d.size}")

    # ======================= Operations =========================================
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

    transform2d_copy = transform2d.copy()
    logger.info(f"Copied Transform2D: {transform2d_copy}")

    transform2d_numpy = transform2d.to_numpy(copy=True)
    logger.info(f"NumPy Transform2D:\n{transform2d_numpy}")

    numpy_array = np.asarray(transform2d)
    total = numpy_array + np.array([1, 1, 0])
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Sum of Transform2D with NumPy array:\n{total}")

    # ======================= Visualize =========================================
    rr.init("transform2d_example", spawn=True)
    datatypes.visualize(transform2d, entity_path="/transform2d", label="Transform2D")

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
