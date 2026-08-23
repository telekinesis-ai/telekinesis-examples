"""Demonstrates the Telekinesis Quaternion datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis import datatypes


def quaternion_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    quaternion = datatypes.Quaternion([0.4619398, 0.1913417, 0.4619398, 0.7325378])
    logger.info(f"Created Quaternion: {quaternion}")

    # ======================= Inspect ===========================================
    logger.info(f"data={quaternion.data}")
    logger.info(f"shape={quaternion.shape}")
    logger.info(f"ndim={quaternion.ndim}")
    logger.info(f"dtype={quaternion.dtype}")
    logger.info(f"size={quaternion.size}")
    logger.info(f"quat_norm_atol={quaternion.quat_norm_atol}")

    # ======================= Operations =========================================
    quaternion.data = [0.0, 0.0, 0.7071068, 0.7071068]
    logger.info(f"Updated Quaternion: {quaternion}")

    quaternion_copy = quaternion.copy()
    logger.info(f"Copied Quaternion: {quaternion_copy}")

    quaternion_numpy = quaternion.to_numpy(copy=True)
    logger.info(f"NumPy Quaternion: {quaternion_numpy}")

    numpy_array = np.asarray(quaternion)
    logger.info(f"NumPy array: {numpy_array}")

    norm = np.linalg.norm(quaternion)
    logger.info(f"Norm (np.linalg.norm): {norm}")

    # Quaternion.data is scalar-first [qw, qx, qy, qz]; scipy expects scalar-last
    # [qx, qy, qz, qw], so reorder before handing it to Rotation.from_quat.
    qw, qx, qy, qz = quaternion.data
    rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    logger.info(f"Equivalent rotation matrix (scipy Rotation):\n{rotation_matrix}")

    # ======================= Visualize =========================================
    rr.init("quaternion_example", spawn=True)
    datatypes.visualize(quaternion, entity_path="/quaternion", label="Updated Quaternion")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(quaternion)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Quaternion: {deserialized}")
    logger.info(f"Round-trip successful: {quaternion == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    quaternion_example()
