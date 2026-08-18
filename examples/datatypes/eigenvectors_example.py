"""Demonstrates the Telekinesis EigenVectors datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def eigenvectors_example():
    """Demonstrate creation, inspection, visualization, update, tolerance relaxation, eigen-relation verification, and serialization."""

    # ======================= Create ============================================
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    w, v = np.linalg.eigh(matrix)
    eigenvectors = datatypes.EigenVectors(v)

    logger.info(f"Input eigenvectors:\n{v}")
    logger.info(f"Original EigenVectors: {eigenvectors}")

    # ======================= Inspect ===========================================
    data = eigenvectors.data
    shape = eigenvectors.shape
    size = eigenvectors.size
    dtype = eigenvectors.dtype
    ndim = eigenvectors.ndim
    numpy_array = eigenvectors.to_numpy()
    eigenvectors_copy = eigenvectors.copy()

    logger.info(
        f"shape={shape}, "
        f"size={size}, "
        f"ndim={ndim}, "
        f"dtype={dtype}"
    )
    logger.info(f"Data:\n{data}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied EigenVectors: {eigenvectors_copy}")

    # ======================= Visualize =========================================
    rr.init("eigenvectors_example", spawn=True)
    datatypes.visualize(eigenvectors, entity_path="/EigenVectors", label="Original EigenVectors")

    # ======================= Update ============================================
    _, new_v = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float64))
    eigenvectors.data = new_v
    logger.info(f"Updated EigenVectors: {eigenvectors}")
    datatypes.visualize(
        eigenvectors, entity_path="/EigenVectors/updated", label="Updated EigenVectors"
    )

    # ======================= Tolerance =========================================
    relaxed = datatypes.EigenVectors(v + 1e-10, atol=1e-6)
    logger.info(f"Noisy input accepted with atol=1e-6: {relaxed}")

    # ======================= Verify ============================================
    for k in range(eigenvectors.shape[1]):
        v_k = data[:, k]
        lhs = matrix @ v_k
        rhs = w[k] * v_k
        logger.info(f"Eigenvector {k} satisfies A @ v == w * v: {np.allclose(lhs, rhs)}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(eigenvectors)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized EigenVectors: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == eigenvectors}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    eigenvectors_example()
