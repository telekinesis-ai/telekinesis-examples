"""Demonstrates the Telekinesis Covariance6x6 datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def covariance6x6_example():
    """Demonstrate creation, inspection, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    matrix = np.diag([1.0, 2.0, 3.0, 0.5, 0.5, 0.5]).astype(np.float32)
    covariance = datatypes.Covariance6x6(matrix)
    logger.info(f"Input matrix:\n{matrix}")
    logger.info(f"Original Covariance6x6: {covariance}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={covariance.shape}, "
        f"size={covariance.size}, "
        f"ndim={covariance.ndim}, "
        f"dtype={covariance.dtype}"
    )
    logger.info(f"Data:\n{covariance.data}")
    logger.info(f"NumPy array:\n{covariance.to_numpy()}")
    logger.info(f"Copied Covariance6x6: {covariance.copy()}")

    # ======================= Visualize =========================================
    rr.init("covariance6x6_example", spawn=True)
    datatypes.visualize(covariance, entity_path="/Covariance6x6")

    # ======================= Update ============================================
    covariance.data = np.eye(6, dtype=np.float32) * 2.0
    logger.info(f"Updated Covariance6x6:\n{covariance.data}")
    datatypes.visualize(covariance, entity_path="/Covariance6x6/updated")

    # ======================= NumPy Interop =====================================
    is_symmetric = np.allclose(covariance.data, covariance.data.T)
    eigenvalues = np.linalg.eigvalsh(covariance.data)
    variances = np.diag(covariance.data)

    logger.info(f"Is symmetric (np.allclose with transpose): {is_symmetric}")
    logger.info(f"Eigenvalues (np.linalg.eigvalsh): {eigenvalues}")
    logger.info(f"Per-axis variances (np.diag): {variances}")

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
