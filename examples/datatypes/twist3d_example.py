"""Demonstrates the Telekinesis Twist3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def twist3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    values = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    twist3d = datatypes.Twist3D(values)
    logger.info(f"Created Twist3D: {twist3d}")

    twist3d_from_coerce = datatypes.Twist3D.coerce(values)
    logger.info(f"Twist3D created via coerce: {twist3d_from_coerce}")

    # ======================= Inspect ===========================================
    logger.info(f"data={twist3d.data}")
    logger.info(f"shape={twist3d.shape}")
    logger.info(f"ndim={twist3d.ndim}")
    logger.info(f"dtype={twist3d.dtype}")
    logger.info(f"size={twist3d.size}")

    # ======================= Operations =========================================
    twist3d.data = np.array([0.0, 0.3, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    logger.info(f"Updated Twist3D: {twist3d}")

    twist3d_copy = twist3d.copy()
    logger.info(f"Copied Twist3D: {twist3d_copy}")

    twist3d_numpy = twist3d.to_numpy(copy=True)
    logger.info(f"NumPy Twist3D: {twist3d_numpy}")

    numpy_array = np.asarray(twist3d)
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Sum: {np.sum(twist3d)}")
    logger.info(f"length={len(twist3d)}")

    # ======================= Visualize =========================================
    rr.init("twist3d_example", spawn=True)
    datatypes.visualize(
        twist3d_from_coerce, entity_path="/twist3d/coerced", label="Coerced Twist3D"
    )
    datatypes.visualize(twist3d, entity_path="/twist3d/updated", label="Updated Twist3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(twist3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Twist3D: {deserialized}")
    logger.info(f"Round-trip successful: {twist3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    twist3d_example()
