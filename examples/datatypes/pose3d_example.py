"""Demonstrates the Telekinesis Pose3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def pose3d_example():
    """Demonstrate creation, access, visualization, update, transform-matrix conversion, pose-format conversion, NumPy interop, and serialization."""

    # ======================= Create ============================================
    pose_data = [0.5, 0.2, 0.5, 0, 60, 90]
    pose3d = datatypes.Pose3D(pose_data)

    logger.info(f"Original Pose3D: {pose3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={pose3d.shape}, "
        f"size={pose3d.size}, "
        f"ndim={pose3d.ndim}, "
        f"dtype={pose3d.dtype}"
    )
    logger.info(f"Underlying data: {pose3d.data}")
    logger.info(f"NumPy array: {pose3d.to_numpy()}")
    logger.info(f"Copy: {pose3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("pose3d_example", spawn=True)
    datatypes.visualize(pose3d, entity_path="/Pose3D", label="my_pose3d")

    # ======================= Update ============================================
    pose3d.data = [0.1, 0.2, 0.3, 0.0, 0.0, 90]
    logger.info(f"Updated Pose3D: {pose3d}")
    datatypes.visualize(pose3d, entity_path="/Pose3D/updated", label="updated_pose3d")

    # ======================= Transform Matrix ==================================
    matrix = pose3d.as_transformation_matrix()
    logger.info(f"Pose3D as transformation matrix:\n{matrix}")
    transform3d = datatypes.Transform3D(matrix)
    datatypes.visualize(transform3d, entity_path="/Transform3D", label="pose3d_transform")

    pose3d_from_transform = datatypes.Pose3D.from_transformation_matrix(transform3d.data)
    logger.info(f"Pose3D from transformation matrix: {pose3d_from_transform}")
    logger.info(f"Converted back to Pose3D is equal to original: {pose3d_from_transform == pose3d}")

    # ======================= Pose Formats ======================================
    pose3d_from_quat = datatypes.Pose3D.from_quat(
        [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795],
    )
    logger.info(f"Pose3D from quaternion: {pose3d_from_quat}")

    pose3d_from_rad = datatypes.Pose3D.from_euler(
        [0.5, 0.2, 0.8, np.radians(30), np.radians(45), np.radians(60)],
        degrees=False,
    )
    logger.info(f"Pose3D from radians: {pose3d_from_rad}")

    pose3d_from_rotvec = datatypes.Pose3D.from_rotvec(
        [0.5, 0.2, 0.8, 0.0, 0.0, np.pi / 2],
    )
    logger.info(f"Pose3D from rotation vector: {pose3d_from_rotvec}")

    # ======================= Convert To ===========================================
    logger.info(f"Pose3D to rotation in degrees: {pose3d.as_euler(degrees=True)}")
    logger.info(f"Pose3D to rotation in radians: {pose3d.as_euler(degrees=False)}")
    logger.info(f"Pose3D to rotation vector: {pose3d.as_rotvec()}")
    logger.info(f"Pose3D to transformation matrix: {pose3d.as_transformation_matrix()}")

    # ======================= NumPy Interop =====================================
    reshaped = np.reshape(pose3d, (6,))
    logger.info(f"Pose3D with np.reshape: {reshaped}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(pose3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Pose3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == pose3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    pose3d_example()
