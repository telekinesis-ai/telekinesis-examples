"""Demonstrates the Telekinesis Covariance6x6 datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def covariance6x6_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    matrix = np.diag([1.0, 2.0, 3.0, 0.5, 0.5, 0.5]).astype(np.float32)
    covariance = datatypes.Covariance6x6(matrix)
    logger.info(f"Created Covariance6x6:\n{covariance.data}")

    # ======================= Inspect ===========================================
    logger.info(f"data=\n{covariance.data}")
    logger.info(f"shape={covariance.shape}")
    logger.info(f"ndim={covariance.ndim}")
    logger.info(f"dtype={covariance.dtype}")
    logger.info(f"size={covariance.size}")
    logger.info(f"covariance_atol={covariance.covariance_atol}")

    # ======================= Operations =========================================
    covariance_copy = covariance.copy()
    logger.info(f"Copied Covariance6x6:\n{covariance_copy.data}")

    covariance.data = np.eye(6, dtype=np.float32) * 2.0
    logger.info(f"Updated Covariance6x6:\n{covariance.data}")

    covariance_numpy = covariance.to_numpy(copy=True)
    logger.info(f"NumPy Covariance6x6:\n{covariance_numpy}")

    numpy_array = np.asarray(covariance)
    logger.info(f"NumPy array via __array__:\n{numpy_array}")

    is_symmetric = np.allclose(covariance.data, covariance.data.T)
    eigenvalues = np.linalg.eigvalsh(covariance.data)
    variances = np.diag(covariance.data)

    logger.info(f"Is symmetric (np.allclose with transpose): {is_symmetric}")
    logger.info(f"Eigenvalues (np.linalg.eigvalsh): {eigenvalues}")
    logger.info(f"Per-axis variances (np.diag): {variances}")

    # ======================= Visualize =========================================
    rr.init("covariance6x6_example", spawn=True)
    datatypes.visualize(covariance_copy, entity_path="/covariance6x6/original")
    datatypes.visualize(covariance, entity_path="/covariance6x6/updated")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(covariance)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Covariance6x6:\n{deserialized.data}")
    logger.info(f"Round-trip successful: {covariance == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    covariance6x6_example()
