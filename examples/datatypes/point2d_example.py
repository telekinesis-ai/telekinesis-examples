"""Demonstrates the Telekinesis Point2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def point2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    point = [1.0, 2.0]
    point2d = datatypes.Point2D(point)
    logger.info(f"Created Point2D: {point2d}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={point2d.shape}")
    logger.info(f"size={point2d.size}")
    logger.info(f"ndim={point2d.ndim}")
    logger.info(f"dtype={point2d.dtype}")
    logger.info(f"data={point2d.data}")

    # ======================= Operations =========================================
    updated_data = [3.0, 4.0]
    point2d.data = updated_data
    logger.info(f"Updated Point2D: {point2d}")

    point2d_copy = point2d.copy()
    logger.info(f"Copied Point2D: {point2d_copy}")

    point2d_numpy = point2d.to_numpy(copy=False)
    logger.info(f"NumPy Point2D: {point2d_numpy}")

    # Translate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0]
    translated_data = point2d.data + np.asarray(translation, dtype=np.float32)
    translated_point2d = datatypes.Point2D(translated_data)
    logger.info(f"Translated Point2D: {translated_point2d}")

    numpy_point2d = np.asarray(point2d)
    logger.info(f"NumPy array via __array__: {numpy_point2d}")

    # ======================= Visualize =========================================
    rr.init("point2d_example", spawn=True)
    datatypes.visualize(point2d, entity_path="/point2d/updated", label="Updated Point2D")
    datatypes.visualize(
        translated_point2d, entity_path="/point2d/translated", label="Translated Point2D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Point2D: {deserialized}")
    logger.info(f"Round-trip successful: {point2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point2d_example()
