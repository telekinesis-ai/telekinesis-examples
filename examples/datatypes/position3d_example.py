"""Demonstrates the Telekinesis Position3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def position3d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    position3d = datatypes.Position3D([1.0, 2.0, 3.0])
    logger.info(f"Original Position3D: {position3d}")

    # ======================= Inspect ===========================================
    data = position3d.data
    shape = position3d.shape
    size = position3d.size
    dtype = position3d.dtype
    ndim = position3d.ndim
    numpy_array = position3d.to_numpy()
    copy = position3d.copy()

    logger.info(f"data={data}, shape={shape}, size={size}, ndim={ndim}, dtype={dtype}")
    logger.info(f"NumPy array: {numpy_array}")
    logger.info(f"Copied Position3D: {copy}")

    # ======================= Visualize =========================================
    rr.init("position3d_example", spawn=True)
    datatypes.visualize(position3d, entity_path="/Position3D", label="My Position3D")

    # ======================= Update ============================================
    updated_data = [4.0, 5.0, 6.0]
    position3d.data = updated_data
    logger.info(f"Updated Position3D: {position3d}")
    datatypes.visualize(
        position3d, entity_path="/Position3D/updated", label="Updated Position3D"
    )

    # ======================= Arithmetic ========================================
    diff = position3d - np.array([1.0, 1.0, 1.0])
    logger.info(f"Difference of Position3D with numpy array: {diff}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(position3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Position3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == updated_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    position3d_example()
