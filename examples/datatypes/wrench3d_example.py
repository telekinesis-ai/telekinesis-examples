"""Demonstrates the Telekinesis Wrench3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def wrench3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    wrench3d = datatypes.Wrench3D(values)
    logger.info(f"Created Wrench3D: {wrench3d}")

    wrench3d_from_coerce = datatypes.Wrench3D.coerce(values)
    logger.info(f"Wrench3D created via coerce: {wrench3d_from_coerce}")

    # ======================= Inspect ===========================================
    logger.info(f"data={wrench3d.data}")
    logger.info(f"shape={wrench3d.shape}")
    logger.info(f"ndim={wrench3d.ndim}")
    logger.info(f"dtype={wrench3d.dtype}")
    logger.info(f"size={wrench3d.size}")

    # ======================= Operations =========================================
    wrench3d.data = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 1.5], dtype=np.float32)
    logger.info(f"Updated Wrench3D: {wrench3d}")

    wrench3d_copy = wrench3d.copy()
    logger.info(f"Copied Wrench3D: {wrench3d_copy}")

    wrench3d_numpy = wrench3d.to_numpy(copy=True)
    logger.info(f"NumPy Wrench3D: {wrench3d_numpy}")

    numpy_array = np.asarray(wrench3d)
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Sum: {np.sum(wrench3d)}")
    logger.info(f"length={len(wrench3d)}")

    # ======================= Visualize =========================================
    rr.init("wrench3d_example", spawn=True)
    datatypes.visualize(
        wrench3d_from_coerce, entity_path="/wrench3d/coerced", label="Coerced Wrench3D"
    )
    datatypes.visualize(wrench3d, entity_path="/wrench3d/updated", label="Updated Wrench3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(wrench3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Wrench3D: {deserialized}")
    logger.info(f"Round-trip successful: {wrench3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    wrench3d_example()
