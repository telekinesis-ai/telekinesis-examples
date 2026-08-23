"""Demonstrates the Telekinesis Pose3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def pose3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    pose_data = [0.5, 0.2, 0.5, 0.0, 60.0, 90.0]
    pose3d = datatypes.Pose3D(pose_data)
    logger.info(f"Created Pose3D: {pose3d}")

    pose3d_from_quat = datatypes.Pose3D.from_quat(
        [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795]
    )
    logger.info(f"Pose3D created from quaternion: {pose3d_from_quat}")

    pose3d_from_euler = datatypes.Pose3D.from_euler(
        [0.5, 0.2, 0.8, np.radians(30), np.radians(45), np.radians(60)],
        degrees=False,
    )
    logger.info(f"Pose3D created from radians: {pose3d_from_euler}")

    pose3d_from_rotvec = datatypes.Pose3D.from_rotvec([0.5, 0.2, 0.8, 0.0, 0.0, np.pi / 2])
    logger.info(f"Pose3D created from rotation vector: {pose3d_from_rotvec}")

    pose3d_from_matrix = datatypes.Pose3D.from_transformation_matrix(
        np.eye(4, dtype=np.float32)
    )
    logger.info(f"Pose3D created from transformation matrix: {pose3d_from_matrix}")

    # ======================= Inspect ===========================================
    logger.info(f"data={pose3d.data}")
    logger.info(f"shape={pose3d.shape}")
    logger.info(f"ndim={pose3d.ndim}")
    logger.info(f"dtype={pose3d.dtype}")
    logger.info(f"size={pose3d.size}")
    logger.info(f"position={pose3d.position}")
    logger.info(f"orientation={pose3d.orientation}")

    # ======================= Operations =========================================
    pose3d.data = [0.1, 0.2, 0.3, 0.0, 0.0, 90.0]
    logger.info(f"Updated Pose3D: {pose3d}")

    pose3d_copy = pose3d.copy()
    logger.info(f"Copied Pose3D: {pose3d_copy}")

    pose3d_numpy = pose3d.to_numpy(copy=True)
    logger.info(f"NumPy Pose3D: {pose3d_numpy}")

    pose3d_quat = pose3d.as_quat()
    logger.info(f"Pose3D as quaternion: {pose3d_quat}")

    pose3d_euler_deg = pose3d.as_euler(degrees=True)
    logger.info(f"Pose3D as Euler degrees: {pose3d_euler_deg}")

    pose3d_euler_rad = pose3d.as_euler(degrees=False)
    logger.info(f"Pose3D as Euler radians: {pose3d_euler_rad}")

    pose3d_rotvec = pose3d.as_rotvec()
    logger.info(f"Pose3D as rotation vector: {pose3d_rotvec}")

    transformation_matrix = pose3d.as_transformation_matrix()
    logger.info(f"Pose3D as transformation matrix:\n{transformation_matrix}")

    transform3d = pose3d.to_transform3d()
    logger.info(f"Pose3D as Transform3D: {transform3d}")

    reshaped = np.reshape(pose3d, (6,))
    logger.info(f"Pose3D with np.reshape: {reshaped}")

    # ======================= Visualize =========================================
    rr.init("pose3d_example", spawn=True)
    datatypes.visualize(pose3d, entity_path="/pose3d", label="Pose3D")
    datatypes.visualize(
        transform3d, entity_path="/pose3d/transform3d", label="Pose3D As Transform3D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(pose3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Pose3D: {deserialized}")
    logger.info(f"Round-trip successful: {pose3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    pose3d_example()
