"""Demonstrates the Telekinesis Twist3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def twist3d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    values = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    twist3d = datatypes.Twist3D(values)

    logger.info(f"Input values: {values}")
    logger.info(f"Created Twist3D: {twist3d}")

    # ======================= Inspect ===========================================
    data = twist3d.data
    shape = twist3d.shape
    size = twist3d.size
    dtype = twist3d.dtype
    ndim = twist3d.ndim
    numpy_array = twist3d.to_numpy()
    twist3d_copy = twist3d.copy()

    logger.info(f"shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"Twist3D data: {data}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied Twist3D: {twist3d_copy}")

    # ======================= Visualize =========================================
    rr.init("twist3d_example", spawn=True)
    datatypes.visualize(twist3d, entity_path="/Twist3D/main", label="Original Twist3D")

    # ======================= Update ============================================
    twist3d.data = np.array([0.0, 0.3, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)

    logger.info(f"Updated Twist3D: {twist3d}")
    datatypes.visualize(twist3d, entity_path="/Twist3D/updated", label="Updated Twist3D")

    # ======================= NumPy Interop =====================================
    linear = numpy_array[:3]
    angular = numpy_array[3:]
    linear_speed = np.linalg.norm(linear)
    angular_speed = np.linalg.norm(angular)

    logger.info(f"Linear velocity (vx, vy, vz): {linear}")
    logger.info(f"Angular velocity (wx, wy, wz): {angular}")
    logger.info(f"linear_speed={linear_speed}, angular_speed={angular_speed}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(twist3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Twist3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == twist3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    twist3d_example()
