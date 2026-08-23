"""Demonstrates the Telekinesis Transform3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def transform3d_example():
    """Demonstrate creation, access, update, inverse, pose conversion, and serialization."""

    # ======================= Create ============================================
    matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1],
            [0.8535534, 0.1464466, -0.5000000, 2],
            [0.1464466, 0.8535534, 0.5000000, 3],
            [0, 0, 0, 1],
        ]
    )
    transform3d = datatypes.Transform3D(matrix)
    logger.info(f"Created Transform3D: {transform3d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={transform3d.shape}, "
        f"size={transform3d.size}, "
        f"ndim={transform3d.ndim}, "
        f"dtype={transform3d.dtype}"
    )
    logger.info(f"Transform3D data:\n{transform3d.data}")
    logger.info(f"NumPy array:\n{ transform3d.to_numpy()}")
    logger.info(f"Copied Transform3D: {transform3d.copy()}")

    # ======================= Visualize =========================================
    rr.init("transform3d_example", spawn=True)
    datatypes.visualize(transform3d, entity_path="/Transform3D/main", label="transform3d")

    # ======================= Update ============================================
    new_matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1.5],
            [0.8535534, 0.1464466, -0.5000000, 2.5],
            [0.1464466, 0.8535534, 0.5000000, 3],
            [0, 0, 0, 1],
        ]
    )
    transform3d.data = new_matrix

    logger.info(f"Updated Transform3D: {transform3d}")
    datatypes.visualize(
        transform3d, entity_path="/Transform3D/updated", label="updated_transform3d"
    )

    # ======================= Inverse ===========================================
    inverse = transform3d.inverse()

    logger.info(f"Inverse Transform3D: {inverse}")
    datatypes.visualize(inverse, entity_path="/Transform3D/inverse", label="inverse_transform3d")

    # ======================= Pose Conversion ===================================
    pose3d = transform3d.to_pose3d()
    pose_quat = pose3d.as_quat()

    logger.info(f"Pose (deg): {pose3d.as_euler(degrees=True)}")
    logger.info(f"Pose (rotvec): {pose3d.as_rotvec()}")
    logger.info(f"Pose (rad): {pose3d.as_euler(degrees=False)}")
    logger.info(f"Pose (quat): {pose_quat}")

    new_transform3d = datatypes.Transform3D.from_pose(pose_quat)

    logger.info(f"New Transform3D from pose: {new_transform3d}")
    logger.info(
        f"Transformation error: {transform3d.compute_transformation_error(new_transform3d)}"
    )

    # ======================= NumPy Interop =====================================
    sum_result = np.array([1, 1, 1, 0]) + transform3d
    logger.info(f"Sum of Transform3D with numpy array: {sum_result}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(transform3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Transform3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == transform3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    transform3d_example()
