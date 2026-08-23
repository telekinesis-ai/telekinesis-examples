"""Demonstrates the Telekinesis Eigenvectors datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def eigenvectors_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    _, eigenvector_data = np.linalg.eigh(matrix)
    eigenvectors = datatypes.Eigenvectors(eigenvector_data)
    logger.info(f"Created Eigenvectors: {eigenvectors}")

    noisy_eigenvectors = datatypes.Eigenvectors(eigenvector_data + 1e-10, atol=1e-6)
    logger.info(f"Created Eigenvectors with relaxed tolerance: {noisy_eigenvectors}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={eigenvectors.shape}")
    logger.info(f"size={eigenvectors.size}")
    logger.info(f"ndim={eigenvectors.ndim}")
    logger.info(f"dtype={eigenvectors.dtype}")
    logger.info(f"data={eigenvectors.data}")

    # ======================= Operations =========================================
    _, new_eigenvector_data = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float64))
    eigenvectors.data = new_eigenvector_data
    logger.info(f"Updated Eigenvectors: {eigenvectors}")

    eigenvectors_copy = eigenvectors.copy()
    logger.info(f"Copied Eigenvectors: {eigenvectors_copy}")

    eigenvectors_numpy = eigenvectors.to_numpy(copy=True)
    logger.info(f"NumPy Eigenvectors:\n{eigenvectors_numpy}")

    numpy_eigenvectors = np.asarray(eigenvectors)
    logger.info(f"NumPy array via __array__:\n{numpy_eigenvectors}")

    logger.info(f"is_orthonormal={eigenvectors.is_orthonormal()}")
    logger.info(f"check_orthonormality={eigenvectors.check_orthonormality(eigenvectors.data)}")

    # ======================= Visualize =========================================
    rr.init("eigenvectors_example", spawn=True)
    datatypes.visualize(eigenvectors, entity_path="/eigenvectors")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(eigenvectors)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Eigenvectors: {deserialized}")
    logger.info(f"Round-trip successful: {eigenvectors == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    eigenvectors_example()
