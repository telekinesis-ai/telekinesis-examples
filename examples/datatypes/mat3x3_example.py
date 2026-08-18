"""Demonstrates the Telekinesis Mat3x3 datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def mat3x3_example():
    """Demonstrate creation, access, copying, visualization, update, and serialization."""

    # ======================= Create ============================================
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    mat = datatypes.Mat3x3(matrix)

    logger.info(f"Created Mat3x3: {mat}")

    # ======================= Inspect ===========================================
    mat_copy = mat.copy()

    logger.info(f"shape={mat.shape}, size={mat.size}, dtype={mat.dtype}, ndim={mat.ndim}")
    logger.info(f"Mat3x3 data: {mat.data}")
    logger.info(f"NumPy array: {mat.to_numpy()}")
    logger.info(f"Copied Mat3x3: {mat_copy}")

    # ======================= Visualize =========================================
    rr.init("mat3x3_example", spawn=True)
    datatypes.visualize(mat, entity_path="/Mat3x3")

    # ======================= Update ============================================
    updated_data = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    mat.data = updated_data

    logger.info(f"Updated Mat3x3: {mat}")
    datatypes.visualize(mat, entity_path="/Mat3x3/updated")

    # ======================= NumPy Interop =====================================
    summed = mat + np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    logger.info(f"Sum with NumPy array: {summed}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mat)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mat3x3: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == updated_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mat3x3_example()
