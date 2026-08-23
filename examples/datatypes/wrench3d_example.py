"""Demonstrates the Telekinesis Wrench3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def wrench3d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    wrench3d = datatypes.Wrench3D(values)

    logger.info(f"Input values: {values}")
    logger.info(f"Created Wrench3D: {wrench3d}")

    # ======================= Inspect ===========================================
    numpy_array = wrench3d.to_numpy()

    logger.info(
        f"shape={wrench3d.shape}, "
        f"size={wrench3d.size}, "
        f"ndim={wrench3d.ndim}, "
        f"dtype={wrench3d.dtype}"
    )
    logger.info(f"Wrench3D data: {wrench3d.data}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied Wrench3D: {wrench3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("wrench3d_example", spawn=True)
    datatypes.visualize(wrench3d, entity_path="/Wrench3D", label="Original Wrench3D")

    # ======================= Update ============================================
    wrench3d.data = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 1.5], dtype=np.float32)

    logger.info(f"Updated Wrench3D: {wrench3d}")
    datatypes.visualize(wrench3d, entity_path="/Wrench3D/updated", label="Updated Wrench3D")

    # ======================= NumPy Interop =====================================
    force = numpy_array[:3]
    torque = numpy_array[3:]
    force_magnitude = np.linalg.norm(force)
    torque_magnitude = np.linalg.norm(torque)

    logger.info(f"Force (fx, fy, fz): {force}")
    logger.info(f"Torque (tx, ty, tz): {torque}")
    logger.info(f"force_magnitude={force_magnitude}, torque_magnitude={torque_magnitude}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(wrench3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Wrench3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == wrench3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    wrench3d_example()
