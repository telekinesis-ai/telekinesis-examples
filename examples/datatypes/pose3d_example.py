"""Demonstrates the Telekinesis Pose3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

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
    matrix = pose3d.to_transformation_matrix()
    logger.info(f"Pose3D as transformation matrix:\n{matrix}")
    transform3d = datatypes.Transform3D(matrix)
    datatypes.visualize(transform3d, entity_path="/Transform3D", label="pose3d_transform")

    pose3d_from_transform = datatypes.Pose3D.from_transformation_matrix(transform3d.data)
    logger.info(f"Pose3D from transformation matrix: {pose3d_from_transform}")
    logger.info(f"Converted back to Pose3D is equal to original: {pose3d_from_transform == pose3d}")

    # ======================= Pose Formats ======================================
    pose3d_deg = datatypes.Pose3D.from_format(
        [30, 45, 60, 0.4619398, 0.1913417, 0.4619398, 0.7325378], rot_type="QUATERNION"
    )
    logger.info(f"Pose3D from pose with rotation in degrees: {pose3d_deg}")

    pose3d_rad = datatypes.Pose3D.from_format(
        [
            np.radians(30),
            np.radians(45),
            np.radians(60),
            0.4619398,
            0.1913417,
            0.4619398,
            0.7325378,
        ],
        rot_type="RADIANS",
    )
    logger.info(f"Pose3D from pose with rotation in radians: {pose3d_rad}")

    pose3d_rotvec = datatypes.Pose3D.from_pose_format(
        [0.5235988, 0.7853982, 1.0471976, 0.4619398, 0.1913417, 0.4619398, 0.7325378],
        rot_type="ROTATION_VECTOR",
    )
    logger.info(f"Pose3D from pose with rotation as rotation vector: {pose3d_rotvec}")

    # ======================= Convert ===========================================
    logger.info(f"Pose3D as rotation in degrees: {pose3d.convert_pose_format(rot_type='deg')}")
    logger.info(f"Pose3D as rotation in radians: {pose3d.convert_pose_format(rot_type='rad')}")
    logger.info(
        f"Pose3D as rotation vector: {pose3d.convert_pose_format(rot_type='rotvec')}"
    )

    # ======================= NumPy Interop =====================================
    reshaped = np.reshape(pose3d, (7,))
    logger.info(f"Underlying Pose3D with np.reshape: {reshaped}")

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
