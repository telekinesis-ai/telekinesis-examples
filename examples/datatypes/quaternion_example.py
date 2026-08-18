"""Demonstrates the Telekinesis Quaternion datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr
from scipy.spatial.transform import Rotation

from telekinesis import datatypes


def quaternion_example():
    """Demonstrate creation, access, visualization, update, NumPy/SciPy interop, and serialization."""

    # ======================= Create ============================================
    quaternion = datatypes.Quaternion([0.4619398, 0.1913417, 0.4619398, 0.7325378])
    logger.info(f"Original Quaternion: {quaternion}")

    # ======================= Inspect ===========================================
    data = quaternion.data
    shape = quaternion.shape
    size = quaternion.size
    dtype = quaternion.dtype
    ndim = quaternion.ndim
    numpy_array = quaternion.to_numpy()
    copy = quaternion.copy()

    logger.info(f"data={data}, shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied Quaternion: {copy}")

    # ======================= Visualize =========================================
    rr.init("quaternion_example", spawn=True)
    datatypes.visualize(quaternion, entity_path="/Quaternion", label="My Quaternion")

    # ======================= Update ============================================
    quaternion.data = [0.0, 0.0, 0.7071068, 0.7071068]
    logger.info(f"Updated Quaternion: {quaternion}")
    datatypes.visualize(
        quaternion, entity_path="/Quaternion/updated", label="Updated Quaternion"
    )

    # ======================= NumPy Interop =====================================
    norm = np.linalg.norm(quaternion.data)
    rotation_matrix = Rotation.from_quat(quaternion.data).as_matrix()

    logger.info(f"Quaternion norm (np.linalg.norm): {norm}")
    logger.info(f"Equivalent rotation matrix (scipy Rotation):\n{rotation_matrix}")

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
