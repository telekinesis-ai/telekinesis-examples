"""Demonstrates the Telekinesis Transform3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def transform3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1.0],
            [0.8535534, 0.1464466, -0.5000000, 2.0],
            [0.1464466, 0.8535534, 0.5000000, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transform3d = datatypes.Transform3D(matrix)
    logger.info(f"Created Transform3D: {transform3d}")

    transform3d_from_pose = datatypes.Transform3D.from_pose(
        [0.5, 0.2, 0.8, 0.0, 0.0, 0.3826834, 0.9238795]
    )
    logger.info(f"Transform3D created from pose: {transform3d_from_pose}")

    # ======================= Inspect ===========================================
    logger.info(f"data=\n{transform3d.data}")
    logger.info(f"shape={transform3d.shape}")
    logger.info(f"ndim={transform3d.ndim}")
    logger.info(f"dtype={transform3d.dtype}")
    logger.info(f"size={transform3d.size}")

    # ======================= Operations =========================================
    new_matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1.5],
            [0.8535534, 0.1464466, -0.5000000, 2.5],
            [0.1464466, 0.8535534, 0.5000000, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transform3d.data = new_matrix
    logger.info(f"Updated Transform3D: {transform3d}")

    transform3d_copy = transform3d.copy()
    logger.info(f"Copied Transform3D: {transform3d_copy}")

    transform3d_numpy = transform3d.to_numpy(copy=True)
    logger.info(f"NumPy Transform3D:\n{transform3d_numpy}")

    inverse_transform3d = transform3d.inverse()
    logger.info(f"Inverse Transform3D: {inverse_transform3d}")

    pose3d = transform3d.to_pose3d()
    logger.info(f"Transform3D as Pose3D: {pose3d}")

    transformation_error = transform3d.compute_transformation_error(transform3d_from_pose)
    logger.info(f"Transformation error vs pose-based Transform3D: {transformation_error}")

    numpy_array = np.asarray(transform3d)
    sum_result = numpy_array + np.array([1, 1, 1, 0])
    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Sum of Transform3D with NumPy array:\n{sum_result}")

    # ======================= Visualize =========================================
    rr.init("transform3d_example", spawn=True)
    datatypes.visualize(transform3d, entity_path="/transform3d", label="Transform3D")
    datatypes.visualize(
        inverse_transform3d, entity_path="/transform3d/inverse", label="Inverse Transform3D"
    )
    datatypes.visualize(
        transform3d_from_pose,
        entity_path="/transform3d/from_pose",
        label="Transform3D From Pose",
    )

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
