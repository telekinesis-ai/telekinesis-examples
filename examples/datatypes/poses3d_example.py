"""Demonstrates the Telekinesis Poses3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def poses3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    poses_data = [[0.5, 0.2, 0.5, 0.0, 60.0, 90.0], [0.1, 0.2, 0.3, 0.0, 0.0, 90.0]]
    poses3d = datatypes.Poses3D(poses_data)
    logger.info(f"Created Poses3D: {poses3d}")

    poses3d_from_quat = datatypes.Poses3D.from_quat(
        [
            [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    logger.info(f"Poses3D created from quaternion: {poses3d_from_quat}")

    poses3d_from_euler = datatypes.Poses3D.from_euler(
        [
            [0.5, 0.2, 0.8, np.radians(30), np.radians(45), np.radians(60)],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ],
        degrees=False,
    )
    logger.info(f"Poses3D created from radians: {poses3d_from_euler}")

    poses3d_from_rotvec = datatypes.Poses3D.from_rotvec(
        [
            [0.5, 0.2, 0.8, 0.0, 0.0, np.pi / 2],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ]
    )
    logger.info(f"Poses3D created from rotation vector: {poses3d_from_rotvec}")

    # ======================= Inspect ===========================================
    logger.info(f"data={poses3d.data}")
    logger.info(f"shape={poses3d.shape}")
    logger.info(f"ndim={poses3d.ndim}")
    logger.info(f"dtype={poses3d.dtype}")
    logger.info(f"size={poses3d.size}")
    logger.info(f"positions={poses3d.positions}")
    logger.info(f"orientations={poses3d.orientations}")

    # ======================= Operations =========================================
    poses3d.data = [[0.1, 0.2, 0.3, 0.0, 0.0, 90.0], [0.4, 0.5, 0.6, 0.0, 0.0, 45.0]]
    logger.info(f"Updated Poses3D: {poses3d}")

    poses3d_copy = poses3d.copy()
    logger.info(f"Copied Poses3D: {poses3d_copy}")

    poses3d_numpy = poses3d.to_numpy(copy=True)
    logger.info(f"NumPy Poses3D: {poses3d_numpy}")

    poses3d_quat = poses3d.as_quat()
    logger.info(f"Poses3D as quaternion: {poses3d_quat}")

    poses3d_euler_deg = poses3d.as_euler(degrees=True)
    logger.info(f"Poses3D as Euler degrees: {poses3d_euler_deg}")

    poses3d_euler_rad = poses3d.as_euler(degrees=False)
    logger.info(f"Poses3D as Euler radians: {poses3d_euler_rad}")

    poses3d_rotvec = poses3d.as_rotvec()
    logger.info(f"Poses3D as rotation vector: {poses3d_rotvec}")

    first_pose3d = poses3d[0]
    logger.info(f"First Pose3D via indexing: {first_pose3d}")

    poses3d_subset = poses3d[0:1]
    logger.info(f"Poses3D subset via slicing: {poses3d_subset}")

    reshaped = np.reshape(poses3d, (2, 6))
    logger.info(f"Poses3D with np.reshape: {reshaped}")

    # ======================= Visualize =========================================
    rr.init("poses3d_example", spawn=True)
    datatypes.visualize(poses3d, entity_path="/poses3d", label=["Poses3D 0", "Poses3D 1"])

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(poses3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Poses3D: {deserialized}")
    logger.info(f"Round-trip successful: {poses3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    poses3d_example()
