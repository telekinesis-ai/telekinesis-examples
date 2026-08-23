"""Demonstrates the Telekinesis Point3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def point3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    point = [1.0, 2.0, 3.0]
    point3d = datatypes.Point3D(point)
    logger.info(f"Created Point3D: {point3d}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={point3d.shape}")
    logger.info(f"size={point3d.size}")
    logger.info(f"ndim={point3d.ndim}")
    logger.info(f"dtype={point3d.dtype}")
    logger.info(f"data={point3d.data}")

    # ======================= Operations =========================================
    updated_data = [4.0, 5.0, 6.0]
    point3d.data = updated_data
    logger.info(f"Updated Point3D: {point3d}")

    point3d_copy = point3d.copy()
    logger.info(f"Copied Point3D: {point3d_copy}")

    point3d_numpy = point3d.to_numpy(copy=False)
    logger.info(f"NumPy Point3D: {point3d_numpy}")

    # Translate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0, 1.0]
    translated_data = point3d.data + np.asarray(translation, dtype=np.float32)
    translated_point3d = datatypes.Point3D(translated_data)
    logger.info(f"Translated Point3D: {translated_point3d}")

    numpy_point3d = np.asarray(point3d)
    logger.info(f"NumPy array via __array__: {numpy_point3d}")

    # ======================= Visualize =========================================
    rr.init("point3d_example", spawn=True)
    datatypes.visualize(point3d, entity_path="/point3d/updated", label="Updated Point3D")
    datatypes.visualize(
        translated_point3d, entity_path="/point3d/translated", label="Translated Point3D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Point3D: {deserialized}")
    logger.info(f"Round-trip successful: {point3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point3d_example()
