"""Demonstrates the Telekinesis Box3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def box3d_example():
    """Demonstrate creation, access, update, translation, scaling, NumPy interop, format conversion, and serialization."""

    # ======================= Create ============================================
    coords = [1, 2, 2.5, 5, 3, 5]
    box3d = datatypes.Box3D(coords)
    logger.info(f"Original Box3D: {box3d}")

    # ======================= Inspect ===========================================
    logger.info(f"Box3D data: {box3d.data}")
    logger.info(
        f"shape={box3d.shape}, "
        f"width={box3d.width}, "
        f"height={box3d.height}, "
        f"depth={box3d.depth}, "
        f"volume={box3d.volume}, "
        f"center={box3d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("box3d_example", spawn=True)
    datatypes.visualize(box3d, entity_path="/Box3D/my_box3d", label="Original Box3D")

    # ======================= Update ============================================
    updated_coords = [1, 4, 2.5, 5, 2, 7]
    box3d.data = updated_coords
    logger.info(f"Updated Box3D: {box3d}")
    datatypes.visualize(box3d, entity_path="/Box3D/my_updated_box3d", label="Updated Box3D")

    # ======================= Translate =========================================
    translation = [3, 3, 1]
    translated_box3d = box3d.translate(translation)
    logger.info(f"Translated Box3D: {translated_box3d}")
    datatypes.visualize(
        translated_box3d, entity_path="/Box3D/my_translated_box3d", label="Translated Box3D"
    )

    # ======================= Scale =============================================
    scale_factors = [2, 0.5, 1.5]
    scaled_box3d = box3d.scale(scale_factors)
    logger.info(f"Scaled Box3D: {scaled_box3d}")
    datatypes.visualize(scaled_box3d, entity_path="/Box3D/my_scaled_box3d", label="Scaled Box3D")

    # ======================= NumPy Interop =====================================
    multiply_factor = 1
    scaled_dimensions = np.multiply(box3d, multiply_factor)
    logger.info(
        f"Box3D dimensions multiplied by {multiply_factor} using numpy: {scaled_dimensions}"
    )

    # ======================= Convert ===========================================
    xyzxyz_coords = [1, 2, 2.5, 5, 3, 5]
    box3d_from_xyzxyz = datatypes.Box3D.from_format(xyzxyz_coords, source_format="xyzxyz")
    logger.info(f"Box3D created from xyzxyz format: {box3d_from_xyzxyz}")

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
