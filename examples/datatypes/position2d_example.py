"""Demonstrates the Telekinesis Position2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def position2d_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    position2d = datatypes.Position2D([10.0, 20.0])
    logger.info(f"Original Position2D: {position2d}")

    # ======================= Inspect ===========================================
    logger.info(
        f"data={position2d.data}, "
        f"shape={position2d.shape}, "
        f"size={position2d.size}, "
        f"ndim={position2d.ndim}, "
        f"dtype={position2d.dtype}"
    )
    logger.info(f"NumPy array: {position2d.to_numpy()}")
    logger.info(f"Copied Position2D: {position2d.copy()}")

    # ======================= Visualize =========================================
    rr.init("position2d_example", spawn=True)
    datatypes.visualize(position2d, entity_path="/Position2D", label="My Position2D")

    # ======================= Update ============================================
    updated_data = [30.0, 40.0]
    position2d.data = updated_data
    logger.info(f"Updated Position2D: {position2d}")
    datatypes.visualize(
        position2d, entity_path="/Position2D/updated", label="Updated Position2D"
    )

    # ======================= Arithmetic ========================================
    logger.info(f"Sum of Position2D with numpy array: {position2d + np.array([5.0, 10.0])}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(position2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Position2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized.data == updated_data}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    position2d_example()
