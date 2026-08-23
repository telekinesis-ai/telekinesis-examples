"""Demonstrates the Telekinesis Box3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def box3d_example():
    """Demonstrate creation, access, update, translation, scaling, NumPy interop, format conversion, and serialization."""

    # ======================= Create ============================================
    # Box3D format is CXCYCZWHD = [cx, cy, cz, width, height, depth]
    coords = [1, 2, 2.5, 5, 3, 5]
    box3d = datatypes.Box3D(coords)
    logger.info(f"Original Box3D: {box3d}")

    # ======================= Inspect ===========================================
    logger.info(f"Box3D data: {box3d.data}")
    logger.info(
        f"dtype={box3d.dtype}, "
        f"ndim={box3d.ndim}, "
        f"shape={box3d.shape}, "
        f"dimensions={box3d.dimensions}, "
        f"volume={box3d.volume}, "
        f"center={box3d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("box3d_example", spawn=True)
    datatypes.visualize(box3d, entity_path="/Box3D/box3d", label="Original Box3D")

    # ======================= Update ============================================
    updated_coords = [1, 4, 2.5, 5, 2, 7]
    box3d.data = updated_coords
    logger.info(f"Updated Box3D: {box3d}")
    datatypes.visualize(box3d, entity_path="/Box3D/updated_box3d", label="Updated Box3D")

    # ======================= Alternate Construction =============================
    xyzxyz_coords = [1, 2, 2.5, 5, 3, 5]
    box3d_from_xyzxyz = datatypes.Box3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"Box3D created from xyzxyz format: {box3d_from_xyzxyz}")

    xyzxyz_view = box3d.as_xyzxyz()
    logger.info(f"Box3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_coords = [1, 2, 2.5, 4, 1, 2.5]
    box3d_from_xyzwhd = datatypes.Box3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"Box3D created from xyzwhd format: {box3d_from_xyzwhd}")

    xyzwhd_view = box3d.as_xyzwhd()
    logger.info(f"Box3D converted to xyzwhd format: {xyzwhd_view}")

    # ======================= NumPy Interop =====================================
    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [3, 3, 1]
    translated_data = box3d.data.copy()
    translated_data[:3] += translation
    translated_box3d = datatypes.Box3D(translated_data)
    logger.info(f"Translated Box3D: {translated_box3d}")
    datatypes.visualize(
        translated_box3d, entity_path="/Box3D/translated_box3d", label="Translated Box3D"
    )

    scale_factors = [2, 0.5, 1.5]
    scaled_data = box3d.data.copy()
    scaled_data[3:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_box3d = datatypes.Box3D(scaled_data)
    logger.info(f"Scaled Box3D: {scaled_box3d}")
    datatypes.visualize(scaled_box3d, entity_path="/Box3D/scaled_box3d", label="Scaled Box3D")

    multiply_factor = 1
    scaled_dimensions = np.multiply(box3d, multiply_factor)
    logger.info(
        f"Box3D dimensions multiplied by {multiply_factor} using numpy: {scaled_dimensions}"
    )

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
