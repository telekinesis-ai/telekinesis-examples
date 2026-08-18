"""Demonstrates the Telekinesis EigenValues datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def eigenvalues_example():
    """Demonstrate creation, inspection, visualization, update, positive-definiteness checks, NumPy interop, and serialization."""

    # ======================= Create ============================================
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    w, _ = np.linalg.eigh(matrix)
    eigenvalues = datatypes.EigenValues(w)

    logger.info(f"Input eigenvalues: {w}")
    logger.info(f"Original EigenValues: {eigenvalues}")

    # ======================= Inspect ===========================================
    data = eigenvalues.data
    shape = eigenvalues.shape
    size = eigenvalues.size
    dtype = eigenvalues.dtype
    ndim = eigenvalues.ndim
    numpy_array = eigenvalues.to_numpy()
    eigenvalues_copy = eigenvalues.copy()

    logger.info(
        f"shape={shape}, "
        f"size={size}, "
        f"ndim={ndim}, "
        f"dtype={dtype}"
    )
    logger.info(f"Data: {data}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied EigenValues: {eigenvalues_copy}")

    # ======================= Visualize =========================================
    rr.init("eigenvalues_example", spawn=True)
    datatypes.visualize(eigenvalues, entity_path="/EigenValues", label="Original EigenValues")

    # ======================= Update ============================================
    new_w, _ = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float32))
    eigenvalues.data = new_w
    logger.info(f"Updated EigenValues: {eigenvalues}")
    datatypes.visualize(
        eigenvalues, entity_path="/EigenValues/updated", label="Updated EigenValues"
    )

    # ======================= Checks ============================================
    positive_definite = eigenvalues.is_positive_definite()
    positive_semidefinite = eigenvalues.is_positive_semidefinite()
    condition_number = eigenvalues.condition_number

    logger.info(
        f"is_positive_definite={positive_definite}, "
        f"is_positive_semidefinite={positive_semidefinite}, "
        f"condition_number={condition_number}"
    )

    # ======================= NumPy Interop =====================================
    spectral_radius = np.max(np.abs(eigenvalues))
    logger.info(f"Spectral radius (max |eigenvalue|): {spectral_radius}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(eigenvalues)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized EigenValues: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == eigenvalues}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    eigenvalues_example()
