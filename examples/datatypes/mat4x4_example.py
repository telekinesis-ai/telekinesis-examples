"""Demonstrates the Telekinesis Mat4x4 datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def mat4x4_example():
    """Demonstrate creation, access, copying, visualization, update, and serialization."""

    # ======================= Create ============================================
    matrix = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    mat = datatypes.Mat4x4(matrix)

    logger.info(f"Created Mat4x4: {mat}")

    # ======================= Inspect ===========================================
    mat_copy = mat.copy()

    logger.info(f"shape={mat.shape}, size={mat.size}, dtype={mat.dtype}, ndim={mat.ndim}")
    logger.info(f"Mat4x4 data: {mat.data}")
    logger.info(f"NumPy array: {mat.to_numpy()}")
    logger.info(f"Copied Mat4x4: {mat_copy}")

    # ======================= Visualize =========================================
    rr.init("mat4x4_example", spawn=True)
    datatypes.visualize(mat, entity_path="/Mat4x4")

    # ======================= Update ============================================
    updated_data = [
        [16.0, 15.0, 14.0, 13.0],
        [12.0, 11.0, 10.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0, 1.0],
    ]
    mat.data = updated_data

    logger.info(f"Updated Mat4x4: {mat}")
    datatypes.visualize(mat, entity_path="/Mat4x4/updated")

    # ======================= NumPy Interop =====================================
    summed = mat + np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    logger.info(f"Sum with NumPy array: {summed}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(mat)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Mat4x4: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == updated_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    mat4x4_example()
