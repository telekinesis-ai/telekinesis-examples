"""Demonstrates the Telekinesis Transforms3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def transforms3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix_1 = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1.0],
            [0.8535534, 0.1464466, -0.5000000, 2.0],
            [0.1464466, 0.8535534, 0.5000000, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix_2 = np.array(
        [
            [0.0, -1.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix_3 = np.eye(4)
    transforms3d = datatypes.Transforms3D([matrix_1, matrix_2, matrix_3])
    logger.info(f"Created Transforms3D: {transforms3d}")

    transforms3d_from_pose = datatypes.Transforms3D.from_pose(
        [
            [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    logger.info(f"Transforms3D created from pose: {transforms3d_from_pose}")

    # ======================= Inspect ===========================================
    logger.info(f"data=\n{transforms3d.data}")
    logger.info(f"shape={transforms3d.shape}")
    logger.info(f"ndim={transforms3d.ndim}")
    logger.info(f"dtype={transforms3d.dtype}")
    logger.info(f"size={transforms3d.size}")

    # ======================= Operations =========================================
    transforms3d_copy = transforms3d.copy()
    logger.info(f"Copied Transforms3D: {transforms3d_copy}")

    transforms3d_numpy = transforms3d.to_numpy(copy=True)
    logger.info(f"NumPy Transforms3D:\n{transforms3d_numpy}")

    inverse_transforms3d = transforms3d.inverse()
    logger.info(f"Inverse Transforms3D: {inverse_transforms3d}")

    poses3d = transforms3d.to_poses3d()
    logger.info(f"Transforms3D as Poses3D: {poses3d}")

    first_transform3d = transforms3d[0]
    logger.info(f"First Transform3D via indexing: {first_transform3d}")

    transforms3d_subset = transforms3d[0:2]
    logger.info(f"Transforms3D subset via slicing: {transforms3d_subset}")

    numpy_array = np.asarray(transforms3d)
    logger.info(f"NumPy array:\n{numpy_array}")

    # ======================= Visualize =========================================
    rr.init("transforms3d_example", spawn=True)
    datatypes.visualize(
        transforms3d,
        entity_path="/transforms3d",
        label=["Transforms3D 0", "Transforms3D 1", "Transforms3D 2"],
    )
    datatypes.visualize(
        transforms3d_from_pose,
        entity_path="/transforms3d/from_pose",
        label=["From Pose 0", "From Pose 1"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(transforms3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Transforms3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == transforms3d}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    transforms3d_example()
