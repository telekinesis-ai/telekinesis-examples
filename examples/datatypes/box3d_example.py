"""Demonstrates the Telekinesis Box3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def box3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Box3D format is CXCYCZWHD = [cx, cy, cz, width, height, depth]
    coords = [1, 2, 2.5, 5, 3, 5]
    box3d = datatypes.Box3D(coords)
    logger.info(f"Original Box3D: {box3d}")

    xyzxyz_coords = [1.0, 1.5, 2.0, 4.0, 4.5, 5.0]
    box3d_from_xyzxyz = datatypes.Box3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"Box3D created from xyzxyz format: {box3d_from_xyzxyz}")

    xyzwhd_coords = [1.0, 1.5, 2.0, 3.0, 3.0, 3.0]
    box3d_from_xyzwhd = datatypes.Box3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"Box3D created from xyzwhd format: {box3d_from_xyzwhd}")

    # ======================= Inspect ===========================================
    logger.info(f"data={box3d.data}")
    logger.info(f"dtype={box3d.dtype}")
    logger.info(f"ndim={box3d.ndim}")
    logger.info(f"shape={box3d.shape}")
    logger.info(f"size={box3d.size}")
    logger.info(f"dimensions={box3d.dimensions}")
    logger.info(f"volume={box3d.volume}")
    logger.info(f"center={box3d.center}")

    # ======================= Operations =========================================
    updated_coords = [1, 4, 2.5, 5, 2, 7]
    box3d.data = updated_coords
    logger.info(f"Updated Box3D: {box3d}")

    xyzxyz_view = box3d.as_xyzxyz()
    logger.info(f"Box3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_view = box3d.as_xyzwhd()
    logger.info(f"Box3D converted to xyzwhd format: {xyzwhd_view}")

    box3d_copy = box3d.copy()
    logger.info(f"Copied Box3D: {box3d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    box3d_numpy = box3d.to_numpy(copy=False)
    logger.info(f"NumPy Box3D:\n{box3d_numpy}")

    numpy_box3d = np.asarray(box3d)
    logger.info(f"Box3D via __array__:\n{numpy_box3d}")

    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [3, 3, 1]
    translated_data = box3d.data.copy()
    translated_data[:3] += translation
    translated_box3d = datatypes.Box3D(translated_data)
    logger.info(f"Translated Box3D: {translated_box3d}")

    scale_factors = [2, 0.5, 1.5]
    scaled_data = box3d.data.copy()
    scaled_data[3:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_box3d = datatypes.Box3D(scaled_data)
    logger.info(f"Scaled Box3D: {scaled_box3d}")

    # ======================= Visualize =========================================
    rr.init("box3d_example", spawn=True)
    datatypes.visualize(box3d, entity_path="/box3d/updated", label="Updated Box3D")
    datatypes.visualize(
        translated_box3d, entity_path="/box3d/translated", label="Translated Box3D"
    )
    datatypes.visualize(scaled_box3d, entity_path="/box3d/scaled", label="Scaled Box3D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(box3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Box3D: {deserialized}")
    logger.info(f"Round-trip successful: {box3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    box3d_example()
