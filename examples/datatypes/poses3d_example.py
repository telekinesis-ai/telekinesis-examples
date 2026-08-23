"""Demonstrates the Telekinesis Poses3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def poses3d_example():
    """Demonstrate creation, access, visualization, update, pose-format conversion, NumPy interop, and serialization."""

    # ======================= Create ============================================
    poses_data = [[0.5, 0.2, 0.5, 0, 60, 90], [0.1, 0.2, 0.3, 0, 0, 90]]
    poses3d = datatypes.Poses3D(poses_data)

    logger.info(f"Original Poses3D: {poses3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={poses3d.shape}, "
        f"size={poses3d.size}, "
        f"ndim={poses3d.ndim}, "
        f"dtype={poses3d.dtype}"
    )
    logger.info(f"Underlying data: {poses3d.data}")
    logger.info(f"NumPy array: {poses3d.to_numpy()}")
    logger.info(f"Copy: {poses3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("poses3d_example", spawn=True)
    datatypes.visualize(poses3d, entity_path="/Poses3D", label=["My Poses3D 0", "My Poses3D 1"])

    # ======================= Update ============================================
    poses3d.data = [[0.1, 0.2, 0.3, 0.0, 0.0, 90], [0.4, 0.5, 0.6, 0.0, 0.0, 45]]
    logger.info(f"Updated Poses3D: {poses3d}")
    datatypes.visualize(
        poses3d,
        entity_path="/Poses3D/updated",
        label=["Updated Poses3D 0", "Updated Poses3D 1"],
    )

    # ======================= Pose Formats ======================================
    poses3d_from_quat = datatypes.Poses3D.from_quat(
        [
            [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    logger.info(f"Poses3D from quaternion: {poses3d_from_quat}")

    poses3d_from_rad = datatypes.Poses3D.from_euler(
        [
            [0.5, 0.2, 0.8, np.radians(30), np.radians(45), np.radians(60)],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ],
        degrees=False,
    )
    logger.info(f"Poses3D from radians: {poses3d_from_rad}")

    poses3d_from_rotvec = datatypes.Poses3D.from_rotvec(
        [
            [0.5, 0.2, 0.8, 0.0, 0.0, np.pi / 2],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ],
    )
    logger.info(f"Poses3D from rotation vector: {poses3d_from_rotvec}")

    # ======================= Convert To ===========================================
    logger.info(f"Poses3D to rotation in degrees: {poses3d.as_euler(degrees=True)}")
    logger.info(f"Poses3D to rotation in radians: {poses3d.as_euler(degrees=False)}")
    logger.info(f"Poses3D to rotation vector: {poses3d.as_rotvec()}")

    # ======================= NumPy Interop =====================================
    reshaped = np.reshape(poses3d, (2, 6))
    logger.info(f"Poses3D with np.reshape: {reshaped}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(poses3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Poses3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == poses3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    poses3d_example()
