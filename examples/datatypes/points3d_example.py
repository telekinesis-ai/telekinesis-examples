"""Demonstrates the Telekinesis Points3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def points3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    points = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    points3d = datatypes.Points3D(points)
    logger.info(f"Created Points3D: {points3d}")

    empty_points3d = datatypes.Points3D(np.empty((0, 3), dtype=np.float32))
    logger.info(f"Created empty Points3D: {empty_points3d}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={points3d.shape}")
    logger.info(f"size={points3d.size}")
    logger.info(f"ndim={points3d.ndim}")
    logger.info(f"dtype={points3d.dtype}")
    logger.info(f"data={points3d.data}")

    # ======================= Operations =========================================
    updated_data = [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]
    points3d.data = updated_data
    logger.info(f"Updated Points3D: {points3d}")

    points3d_copy = points3d.copy()
    logger.info(f"Copied Points3D: {points3d_copy}")

    points3d_numpy = points3d.to_numpy(copy=False)
    logger.info(f"NumPy Points3D:\n{points3d_numpy}")

    # Translate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0, 1.0]
    translated_data = points3d.data + np.asarray(translation, dtype=np.float32)
    translated_points3d = datatypes.Points3D(translated_data)
    logger.info(f"Translated Points3D: {translated_points3d}")

    numpy_points3d = np.asarray(points3d)
    logger.info(f"NumPy array via __array__:\n{numpy_points3d}")

    # ======================= Visualize =========================================
    rr.init("points3d_example", spawn=True)
    datatypes.visualize(
        points3d, entity_path="/points3d/updated", label=["Updated Point 1", "Updated Point 2"]
    )
    datatypes.visualize(
        translated_points3d,
        entity_path="/points3d/translated",
        label=["Translated Point 1", "Translated Point 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(points3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Points3D: {deserialized}")
    logger.info(f"Round-trip successful: {points3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    points3d_example()
