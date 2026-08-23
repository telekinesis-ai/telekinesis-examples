"""Demonstrates the Telekinesis Eigenvalues datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def eigenvalues_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    eigenvalue_data, _ = np.linalg.eigh(matrix)
    eigenvalues = datatypes.Eigenvalues(eigenvalue_data)
    logger.info(f"Created Eigenvalues: {eigenvalues}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={eigenvalues.shape}")
    logger.info(f"size={eigenvalues.size}")
    logger.info(f"ndim={eigenvalues.ndim}")
    logger.info(f"dtype={eigenvalues.dtype}")
    logger.info(f"data={eigenvalues.data}")
    logger.info(f"condition_number={eigenvalues.condition_number}")

    # ======================= Operations =========================================
    new_eigenvalue_data, _ = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float32))
    eigenvalues.data = new_eigenvalue_data
    logger.info(f"Updated Eigenvalues: {eigenvalues}")

    eigenvalues_copy = eigenvalues.copy()
    logger.info(f"Copied Eigenvalues: {eigenvalues_copy}")

    eigenvalues_numpy = eigenvalues.to_numpy(copy=True)
    logger.info(f"NumPy Eigenvalues: {eigenvalues_numpy}")

    numpy_eigenvalues = np.asarray(eigenvalues)
    logger.info(f"NumPy array via __array__: {numpy_eigenvalues}")
    logger.info(f"Spectral radius (max |eigenvalue|): {np.max(np.abs(eigenvalues))}")

    logger.info(f"is_positive_semidefinite={eigenvalues.is_positive_semidefinite()}")
    logger.info(f"is_positive_definite={eigenvalues.is_positive_definite()}")

    # ======================= Visualize =========================================
    rr.init("eigenvalues_example", spawn=True)
    datatypes.visualize(eigenvalues, entity_path="/eigenvalues")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(eigenvalues)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Eigenvalues: {deserialized}")
    logger.info(f"Round-trip successful: {eigenvalues == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    eigenvalues_example()
