"""Demonstrates the Telekinesis Array datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def array_example():
    """Demonstrate creation, access, NumPy interop, update, and serialization."""

    # ======================= Create ============================================
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    array = datatypes.Array(data)

    logger.info(f"Created Array: {array}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={array.shape}, "
        f"size={array.size}, "
        f"ndim={array.ndim}, "
        f"dtype={array.dtype}"
    )
    logger.info(f"Array data:\n{array.data}")

    # ======================= NumPy Interop =====================================
    numpy_array = np.asarray(array)
    reshaped = np.reshape(array, (2, 6))

    logger.info(f"NumPy array:\n{numpy_array}")
    logger.info(f"Reshaped array:\n{reshaped}")
    logger.info(f"Sum: {np.sum(array)}")

    # ======================= Update ============================================
    array.data = np.arange(24, dtype=np.float32).reshape(4, 6)

    logger.info(f"Updated Array: {array}")

    # ======================= Copy ==============================================
    array_copy = array.copy()
    logger.info(f"Copied Array: {array_copy}")

    # ======================= Visualize =========================================
    rr.init("array_example", spawn=True)
    datatypes.visualize(array, entity_path="/array")

    # ======================= Serialize / Deserialize ============================
    start = time.perf_counter()
    serialized = datatypes.serialize(array)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Array: {deserialized}")
    logger.info(f"Round-trip successful: {array == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    array_example()