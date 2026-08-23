"""Demonstrates the Telekinesis Points2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def points2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    points = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    points2d = datatypes.Points2D(points)
    logger.info(f"Created Points2D: {points2d}")

    empty_points2d = datatypes.Points2D(np.empty((0, 2), dtype=np.float32))
    logger.info(f"Created empty Points2D: {empty_points2d}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={points2d.shape}")
    logger.info(f"size={points2d.size}")
    logger.info(f"ndim={points2d.ndim}")
    logger.info(f"dtype={points2d.dtype}")
    logger.info(f"data={points2d.data}")

    # ======================= Operations =========================================
    updated_data = [[70.0, 80.0], [90.0, 100.0], [110.0, 120.0]]
    points2d.data = updated_data
    logger.info(f"Updated Points2D: {points2d}")

    points2d_copy = points2d.copy()
    logger.info(f"Copied Points2D: {points2d_copy}")

    points2d_numpy = points2d.to_numpy(copy=False)
    logger.info(f"NumPy Points2D:\n{points2d_numpy}")

    # Translate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0]
    translated_data = points2d.data + np.asarray(translation, dtype=np.float32)
    translated_points2d = datatypes.Points2D(translated_data)
    logger.info(f"Translated Points2D: {translated_points2d}")

    numpy_points2d = np.asarray(points2d)
    logger.info(f"NumPy array via __array__:\n{numpy_points2d}")

    # ======================= Visualize =========================================
    rr.init("points2d_example", spawn=True)
    datatypes.visualize(
        points2d,
        entity_path="/points2d/updated",
        label=["Updated Point 1", "Updated Point 2", "Updated Point 3"],
    )
    datatypes.visualize(
        translated_points2d,
        entity_path="/points2d/translated",
        label=["Translated Point 1", "Translated Point 2", "Translated Point 3"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(points2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Points2D: {deserialized}")
    logger.info(f"Round-trip successful: {points2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    points2d_example()
